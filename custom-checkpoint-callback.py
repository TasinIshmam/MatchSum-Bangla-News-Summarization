import os

import torch
from fastNLP import Callback
from fastNLP.core._parallel_utils import _model_contains_inner_module


class CheckPointCallback(Callback):
    def __init__(self, save_path, delete_when_train_finish=True, recovery_fitlog=True):
        r"""
        用于在每个epoch结束的时候保存一下当前的Trainer状态，可以用于恢复之前的运行。使用最近的一个epoch继续训练
        一段示例代码
        Example1::

            >>> callback = CheckPointCallback('chkp.pt')
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        Example2::

            >>> fitlog.set_log_dir('xxx')
            >>> callback = CheckPointCallback('chkp.pt')  # 一定要在set_log_dir下一行就接着CheckPointCallback
            >>> trainer = Trainer(xxx, callback=callback)
            >>> trainer.train()  # 如果训练过程没结束就fail，请直接再次运行即可（请务必保证与上次使用了完全相同的数据与超参数）

        :param str save_path: 将状态保存到哪个位置。需要指定一个具体的路径，比如'checkpoints/chtp.pt'。如果检查到该文件存在，将在
            Trainer开始训练的时候自动从这个Checkpoint处开始运行。
        :param bool delete_when_train_finish: 如果Train正常运行完毕，是否自动删除。删除该文件可以使得路径自动复用。
        :param bool recovery_fitlog: 是否恢复fitlog为对应的log，如果为True请将本Callback放在fitlog.set_log_dir后面一行初始化。
            如果为False，将新建一个log folder否则继续使用之前的。
        """
        super().__init__()
        self.save_path = os.path.abspath(os.path.expanduser(save_path))
        self.delete_when_train_finish = delete_when_train_finish
        self.recover_fitlog = recovery_fitlog
        try:
            import fitlog
        except:
            self.recover_fitlog = False
        if os.path.exists(os.path.expanduser(self.save_path)):
            print("The train will start from the checkpoint saved in {}.".format(self.save_path))
            if self.recover_fitlog:
                states = torch.load(self.save_path)
                if 'fitlog_log_dir' in states:
                    try:
                        import fitlog
                        log_dir = states['fitlog_log_dir']
                        if 'fitlog_save_log_dir' in states:
                            log_dir = states['fitlog_save_log_dir']
                        fitlog.set_log_dir(log_dir, new_log=True)
                    except:
                        print("Fail to recovery the fitlog states.")

    def on_train_begin(self):
        r"""
        当train开始时，且需要恢复上次训练时，会做以下的操作
            (1) 重新加载model权重
            (2) 重新加载optimizer的状态
            (3) 加载当前epoch数
            (4) 加载当前最佳evaluate的性能
            (5) (optional) 自动将fitlog设置到上次log出继续

        :return:
        """
        if os.path.exists(os.path.expanduser(self.save_path)):
            states = torch.load(self.save_path)
            model = self.model
            # if _model_contains_inner_module(model):
            #     model = model.module
            # model.load_state_dict(states['model'])
            self.optimizer.load_state_dict(states['optimizer'])
            self.trainer.epoch = states['epoch'] + 1 # 因为是结束储存的，所以需要从下一个epoch开始
            self.trainer.step = states['step']
            if 'best_dev_epoch' in states:
                self.trainer.best_dev_perf = states['best_dev_perf']
                self.trainer.best_dev_epoch = states['best_dev_epoch']
                self.trainer.best_dev_step = states['best_dev_step']
                self.trainer.best_metric_indicator = states['best_metric_indicator']
            print("Load checkpoint from {}".format(os.path.expanduser(self.save_path)))

    def on_epoch_end(self):
        r"""
        保存状态，使得结果可以被恢复

        :param self:
        :return:
        """
        states = {}
        model = self.model
        if _model_contains_inner_module(model):
            model = model.module
        states['model'] = {name:param.cpu() for name, param in model.state_dict().items()}
        states['optimizer'] = self.optimizer.state_dict()
        states['epoch'] = self.epoch
        states['step'] = self.step
        if self.trainer.best_dev_epoch is not None:
            states['best_dev_epoch'] = self.trainer.best_dev_epoch
            states['best_dev_perf'] = self.trainer.best_dev_perf
            states['best_dev_step'] = self.trainer.best_dev_step
            states['best_metric_indicator'] = self.trainer.best_metric_indicator
        if self.recover_fitlog:
            try:
                import fitlog
                if fitlog._logger._log_dir is not None:
                    states['fitlog_log_dir'] = fitlog._logger._log_dir
                if fitlog._logger._save_log_dir is not None:
                    states['fitlog_save_log_dir'] = fitlog._logger._save_log_dir
            except:
                pass
        torch.save(states, self.save_path)
        print("Checkpoint:{} has been saved in epoch:{}.".format(self.save_path, self.epoch))

    def on_train_end(self):
        # 训练结束，根据情况删除保存的内容
        if self.delete_when_train_finish:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
                print("Checkpoint:{} has been removed.".format(self.save_path))
