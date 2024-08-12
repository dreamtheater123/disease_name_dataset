import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from transformers import TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl, get_scheduler
from datasets import load_metric

import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def acc_correction(acc, args, len_y_pred):
    """
    把acc指标进行修正
    :return:
    """
    if args['dataset']['name'] in ['CHIP-CDN', 'CHIP-CDN_EDA', 'CHIP-CDN_BT', 'yidu-n7k']:
        acc = (acc * len_y_pred) / args['dataset']['devset_len']
        return acc
    else:
        return acc


class NativePtTrainer:
    def __init__(self, args, model, datasets):
        # 这里不要用self.args，等train的时候往函数里面传args参数，保证args是唯一的
        # TODO: 写好fine-tune框架
        self.model = model.to(args['device'])

        if args['fine_tune'] != False:
            if 'checkpoint' in args['fine_tune']:
                model_path = os.path.join('./log', args['pretrain_model_path'], 'model', args['fine_tune'] + '.ckpt')
            else:
                model_path = os.path.join('./log', args['pretrain_model_path'], 'model', args['fine_tune'] + '_model.ckpt')
            print(f'pre-trained model path: {model_path}')
            if args['dataset']['name'] == 'yidu-n7k' or 'similarity' in args['pretrain_model_path']:
                pretrained_dict = torch.load(model_path, map_location=args['device'])
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'pred_fc' not in k}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=args['device']), strict=True)
        self.train_set = datasets[0]
        self.dev_set = datasets[1]
        self.test_set = datasets[2]

        self.train_loader = self.get_dataloader(self.train_set, args['batch_size'], mode='train')
        self.dev_loader = self.get_dataloader(self.dev_set, args['batch_size'], mode='dev')
        self.test_loader = self.get_dataloader(self.test_set, args['batch_size'], mode='test')

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(parameters, lr=args['model']['lr'])
        self.criterion = nn.CrossEntropyLoss()
        self.round2eval = min(args['round2eval'], len(self.train_loader))
        self.loss_train = []
        self.num_batch = -1  # 让第一个batch的编号为0
        self.best_acc = 0
        self.model_path = os.path.join('./log', args['save_model_path'], 'model')
        self.early_stop = 0
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if args['enable_writer']:
            writer_path = os.path.join('./log', args['save_model_path'], 'tensorboard')
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            self.writer = SummaryWriter(log_dir=writer_path)  # tensorboard output

    def get_dataloader(self, dataset, batch_size, mode):
        assert mode in ['train', 'dev', 'test']
        if dataset is not None:
            if mode == 'train':  # shuffle data only for training set
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            return data_loader
        else:
            return None

    def train(self, args):
        self.model.train()
        self.tik = time.time()

        for epoch in range(args['epoch']):
            self.model.train()
            print(f"Epoch [{epoch + 1:} / {args['epoch']:}]")
            for batch_data in tqdm(self.train_loader):
                self.model.train()
                self.num_batch += 1  # 所以第一个batch的值就是1
                data = batch_data[0]
                label = batch_data[1]
                pred = self.model(data)

                self.optimizer.zero_grad()
                pred = pred.squeeze()
                batch_loss = self.criterion(pred, label)
                self.loss_train.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()

                # eval
                if self.num_batch % self.round2eval == 0:
                    self.eval(args)
                # early stop
                if self.early_stop >= args['early_stop_round']:
                    print(f'model training early stopped since no improvements in {args["early_stop_round"]} rounds!')
                    print(f'best acc: {self.best_acc}')
                    with open(os.path.join('./log', args['save_model_path'], 'results.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'best acc: {self.best_acc}\n')
                    return
            # save checkpoint for every epoch
            if args['save_checkpoint']:
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f'checkpoint{epoch+1}.ckpt'))
        print(f'best acc: {self.best_acc}')
        with open(os.path.join('./log', args['save_model_path'], 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f'best acc: {self.best_acc}\n')

    def eval(self, args):
        self.model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            loss_val = []
            for batch_data_val in self.dev_loader:
                data_val = batch_data_val[0]
                label_val = batch_data_val[1]
                pred = self.model(data_val)
                batch_loss_val = self.criterion(pred.squeeze(), label_val)
                loss_val.append(batch_loss_val.item())
                pred = nn.Softmax(dim=1)(pred)
                pred = np.argmax(pred.cpu(), axis=1)
                for item in label_val:
                    y_true.append(item)
                for item in pred:
                    y_pred.append(item)
            acc = accuracy_score(np.array(y_true).astype(np.int), np.array(y_pred).astype(np.int))
            # 修正CHIP-CDN等任务的acc值
            acc = acc_correction(acc, args, len(y_pred))
            self.loss_train = np.mean(self.loss_train)
            loss_val = np.mean(loss_val)

            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'newest_model.ckpt'))
            if acc > self.best_acc:
                self.best_acc = acc
                improve = '*'
                # save best model
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.ckpt'))
                self.early_stop = 0
            else:
                improve = ' '
                self.early_stop += 1

            time_spent = (time.time() - self.tik) / 60.0
            print('# batch:', self.num_batch, '|', 'training loss:', round(self.loss_train, 4), '|', 'val loss:',
                  round(loss_val, 4), '|', 'val_acc:', round(acc, 6), '|', 'best_val_acc:', round(self.best_acc, 6),
                  'time:', round(time_spent, 2), '|', 'stop round:', self.early_stop, '|', 'improve:', improve)
            # TODO: tensorboard
            if args['enable_writer']:
                self.writer.add_scalar('hyperparameters/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.num_batch)
                self.writer.add_scalar('hyperparameters/batch_size', args['batch_size'], self.num_batch)
                self.writer.add_scalar('loss/train', self.loss_train, self.num_batch)
                self.writer.add_scalar('acc/dev_acc', acc, self.num_batch)
                self.writer.add_scalar('acc/best_dev_acc', self.best_acc, self.num_batch)
                # self.writer.add_figure("Confusion matrix", cm_figure, self.num_batch)
            self.loss_train = []


class NativePtHfTrainer:
    def __init__(self, args, model, datasets):
        # 这里不要用self.args，等train的时候往函数里面传args参数，保证args是唯一的
        self.model = model.to(args['device'])

        if args['fine_tune'] != False:
            if 'checkpoint' in args['fine_tune']:
                model_path = os.path.join('./log', args['pretrain_model_path'], 'model', args['fine_tune'] + '.ckpt')
            else:  # load checkpoints
                model_path = os.path.join('./log', args['pretrain_model_path'], 'model', args['fine_tune'] + '_model.ckpt')
            print(f'pre-trained model path: {model_path}')
            if args['dataset']['name'] == 'yidu-n7k' or 'similarity' in args['pretrain_model_path']:
                pretrained_dict = torch.load(model_path, map_location=args['device'])
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=args['device']), strict=True)
        self.train_set = datasets[0]
        self.dev_set = datasets[1]
        self.test_set = datasets[2]

        self.train_set = self.to_torch_dataset(self.train_set)
        self.dev_set = self.to_torch_dataset(self.dev_set)
        self.test_set = self.to_torch_dataset(self.test_set)

        self.train_loader = self.get_dataloader(self.train_set, args['batch_size'], mode='train')
        self.dev_loader = self.get_dataloader(self.dev_set, args['batch_size'], mode='dev')
        self.test_loader = self.get_dataloader(self.test_set, args['batch_size'], mode='test')

        self.optimizer = optim.AdamW(model.parameters(), lr=args['model']['lr'])
        num_training_steps = args['epoch'] * len(self.train_loader)
        self.lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        self.criterion = nn.CrossEntropyLoss()
        self.round2eval = min(args['round2eval'], len(self.train_loader))
        self.loss_train = []
        self.num_batch = -1  # 让第一个batch的编号为0
        self.best_acc = 0
        self.model_path = os.path.join('./log', args['save_model_path'], 'model')
        self.early_stop = 0
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if 'similarity' in args['dataset']['name']:
            for path in [os.path.join(self.model_path, 'huggingface', 'best'),
                         os.path.join(self.model_path, 'huggingface', 'newest')]:
                if not os.path.exists(path):
                    os.makedirs(path)

        if args['enable_writer']:
            writer_path = os.path.join('./log', args['save_model_path'], 'tensorboard')
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            self.writer = SummaryWriter(log_dir=writer_path)  # tensorboard output

    def to_torch_dataset(self, dataset):
        if dataset is not None:
            col_names = dataset.column_names
            remove_cols = []
            for col in col_names:
                if 'text' in col:
                    remove_cols.append(col)
            dataset = dataset.remove_columns(remove_cols)
            dataset = dataset.rename_column("label", "labels")
            dataset.set_format("torch")

        return dataset

    def get_dataloader(self, dataset, batch_size, mode):
        assert mode in ['train', 'dev', 'test']
        if dataset is not None:
            if mode == 'train':  # shuffle data only for training set
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            else:
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            return data_loader
        else:
            return None

    def train(self, args):
        self.model.train()
        self.tik = time.time()

        for epoch in range(args['epoch']):
            self.model.train()
            print(f"Epoch [{epoch + 1:} / {args['epoch']:}]")
            for batch_data in tqdm(self.train_loader):
                self.model.train()
                self.num_batch += 1  # 所以第一个batch的值就是1
                batch_data = {k: v.to(args['device']) for k, v in batch_data.items()}
                # data = batch_data[0]
                # label = batch_data[1]
                # pred = self.model(data)
                outputs = self.model(**batch_data)

                self.optimizer.zero_grad()
                batch_loss = outputs.loss
                # batch_loss = self.criterion(pred, label)
                self.loss_train.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # eval
                if self.num_batch % self.round2eval == 0:
                    self.eval(args)
                # early stop
                if self.early_stop >= args['early_stop_round']:
                    print(f'model training early stopped since no improvements in {args["early_stop_round"]} rounds!')
                    print(f'best acc: {self.best_acc}')
                    with open(os.path.join('./log', args['save_model_path'], 'results.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'best acc: {self.best_acc}\n')
                    return
            # save checkpoint for every epoch
            if args['save_checkpoint']:
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f'checkpoint{epoch + 1}.ckpt'))
        print(f'best acc: {self.best_acc}')
        with open(os.path.join('./log', args['save_model_path'], 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f'best acc: {self.best_acc}\n')

    def eval(self, args):
        self.model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            loss_val = []
            for batch_data_val in tqdm(self.dev_loader):
                batch_data_val = {k: v.to(args['device']) for k, v in batch_data_val.items()}
                outputs = self.model(**batch_data_val)
                batch_loss_val = outputs.loss
                # data_val = batch_data_val[0]
                # label_val = batch_data_val[1]
                # pred = self.model(data_val)
                # batch_loss_val = self.criterion(pred.squeeze(), label_val)
                loss_val.append(batch_loss_val.item())
                logits = outputs.logits
                pred = torch.argmax(logits, dim=-1)
                # pred = nn.Softmax(dim=1)(pred)
                # pred = np.argmax(pred.cpu(), axis=1)
                for item in batch_data_val['labels']:
                    y_true.append(item)
                for item in pred:
                    y_pred.append(item)
                a = 1  # for monitor purpose
            acc = accuracy_score(np.array(y_true).astype(np.int), np.array(y_pred).astype(np.int))
            # 修正CHIP-CDN等任务的acc值
            acc = acc_correction(acc, args, len(y_pred))
            self.loss_train = np.mean(self.loss_train)
            loss_val = np.mean(loss_val)

            torch.save(self.model.state_dict(), os.path.join(self.model_path, 'newest_model.ckpt'))
            # 相似度预训练要通过hf格式保存模型，方便参数不同时加载
            if 'similarity' in args['dataset']['name']:
                self.model.save_pretrained(os.path.join(self.model_path, 'huggingface', 'newest'))

            if acc > self.best_acc:
                self.best_acc = acc
                improve = '*'
                # save best model
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'best_model.ckpt'))
                # 相似度预训练要通过hf格式保存模型，方便参数不同时加载
                if 'similarity' in args['dataset']['name']:
                    self.model.save_pretrained(os.path.join(self.model_path, 'huggingface', 'best'))
                self.early_stop = 0
            else:
                improve = ' '
                self.early_stop += 1

            time_spent = (time.time() - self.tik) / 60.0
            print('# batch:', self.num_batch, '|', 'training loss:', round(self.loss_train, 4), '|', 'val loss:',
                  round(loss_val, 4), '|', 'val_acc:', round(acc, 6), '|', 'best_val_acc:', round(self.best_acc, 6),
                  'time:', round(time_spent, 2), '|', 'stop round:', self.early_stop, '|', 'improve:', improve)  # standard output
            # TODO: tensorboard
            if args['enable_writer']:
                self.writer.add_scalar('hyperparameters/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.num_batch)
                self.writer.add_scalar('hyperparameters/batch_size', args['batch_size'], self.num_batch)
                self.writer.add_scalar('loss/train', self.loss_train, self.num_batch)
                self.writer.add_scalar('acc/dev_acc', acc, self.num_batch)
                self.writer.add_scalar('acc/best_dev_acc', self.best_acc, self.num_batch)
                # self.writer.add_figure("Confusion matrix", cm_figure, self.num_batch)
            self.loss_train = []


class MyHfTrainer:
    """
    my own model trainer based on huggingface-provided pytorch trainer
    """
    def __init__(self, args, model, datasets):
        # 这里不要用self.args，等train的时候往函数里面传args参数，保证args是唯一的
        self.model = model
        self.training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=args['epoch'],
            evaluation_strategy='epoch',  # 无视这个warning，这个没问题
            # eval_steps=1
        )
        self.train_set = datasets[0]
        self.dev_set = datasets[1]
        if args['dataset']['name'] == 'CHIP-CDN':
            self.test_set = datasets[1]
        else:
            self.test_set = datasets[2]

        # self.round2eval = min(args['round2eval'], len(self.train_loader))  # hf模型当前默认每个epoch测试一次
        self.num_batch = -1  # 让第一个batch的编号为0
        self.best_acc = 0
        self.model_path = os.path.join('./log', args['save_model_path'], 'model')
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # if args['enable_writer']:
        #     writer_path = os.path.join('./log', args['save_model_path'], 'tensorboard')
        #     if not os.path.exists(writer_path):
        #         os.makedirs(writer_path)
        #     self.writer = SummaryWriter(log_dir=writer_path)  # tensorboard output

        self.callback = MyHfCallback(args)
        self.metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            # TODO: 修改这个accuracy
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return self.metric.compute(predictions=predictions, references=labels)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_set,
            eval_dataset=self.dev_set,
            compute_metrics=compute_metrics,
            callbacks=[self.callback]
        )

    def train(self, args):
        self.trainer.train()


class MyHfCallback(TrainerCallback):
    def __init__(self, args):
        super(TrainerCallback, self).__init__()
        if args['enable_writer']:
            self.enable_writer = True
            writer_path = os.path.join('./log', args['save_model_path'], 'tensorboard')
            if not os.path.exists(writer_path):
                os.makedirs(writer_path)
            self.writer = SummaryWriter(log_dir=writer_path)  # tensorboard output
        else:
            self.enable_writer = False

    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     print(state)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.enable_writer:
            pass


