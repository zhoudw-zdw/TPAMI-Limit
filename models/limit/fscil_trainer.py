from models.base.fscil_trainer import FSCILTrainer as Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

from .helper import *
from utils import *
from dataloader.data_utils import *
from dataloader.sampler import BasePreserverCategoriesSampler,NewCategoriesSampler
from .Network import MYNET


#copy from acastle.
class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, train_fsl_loader,train_gfsl_loader, testloader = self.get_base_dataloader_meta()
            return trainset, train_fsl_loader,train_gfsl_loader, testloader
        else:
            trainset, trainloader, testloader,train_fsl_loader = self.get_new_dataloader(session)
            return trainset, trainloader, testloader,train_fsl_loader

    def get_base_dataloader_meta(self):
        # sample 60 way 1 shot, 15query.
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True,autoaug=self.args.autoaug)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True,autoaug=self.args.autoaug)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path,autoaug=self.args.autoaug)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index,autoaug=self.args.autoaug)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path,autoaug=self.args.autoaug)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index,autoaug=self.args.autoaug)

        
        train_gfsl_loader = DataLoader(dataset=trainset, 
                                   batch_size=self.args.batch_size_base, 
                                   shuffle=True, 
                                   num_workers=8, 
                                   pin_memory=True) 
        train_sampler = CategoriesSampler(trainset.targets,  len(train_gfsl_loader), self.args.sample_class, self.args.sample_shot)
        train_fsl_loader = DataLoader(dataset=trainset, 
                                    batch_sampler=train_sampler, 
                                    num_workers=8, 
                                    pin_memory=True)  

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return trainset, train_fsl_loader,train_gfsl_loader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False,autoaug=self.args.autoaug)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path,autoaug=self.args.autoaug)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path,autoaug=self.args.autoaug)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)
        
        test_sampler = NewCategoriesSampler(trainset.targets, 1, 5, 5)
        train_fsl_loader = DataLoader(dataset=trainset,
                                    batch_sampler=test_sampler,
                                    num_workers=0,
                                    pin_memory=True)  

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False,autoaug=self.args.autoaug)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new,autoaug=self.args.autoaug)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new,autoaug=self.args.autoaug)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

                  
        return trainset, trainloader, testloader,train_fsl_loader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    
    def get_optimizer_base(self):

        top_para = [v for k,v in self.model.named_parameters() if ('encoder' not in k and 'cls' not in k)] 
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': top_para, 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

               
        return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        self.result_list = [args]

        for session in range(args.start_session, args.sessions):
            if session==0:
                train_set, train_fsl_loader,train_gfsl_loader, testloader = self.get_dataloader(session)
            else:
                train_set, trainloader, testloader,train_fsl_loader = self.get_dataloader(session)
            self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, train_fsl_loader,train_gfsl_loader, optimizer, scheduler, epoch, args)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.module.mode = 'avg_cos'

                    if args.set_no_val: # set no validation
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        tsl, tsa = self.test(self.model, testloader, testloader, args, session)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        print('\n epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                        self.result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                    else:
                        # take the last session's testloader for validation
                        vl, va = self.validation()
                        # save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                          self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        self.result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)

                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                # always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                #self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa = self.test(self.model, testloader, None,args, session)
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                train_fsl_loader.dataset.transform = testloader.dataset.transform
                
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                tsl, tsa = self.test(self.model, testloader,train_fsl_loader, args, session,validation=False)

                # save better model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                self.result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        self.result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        self.result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

    def validation(self):
        with torch.no_grad():
            model = self.model
            session=1
            #for session in range(1, self.args.sessions):
            trainset, trainloader, testloader,train_fsl_loader = self.get_dataloader(session)
            trainloader.dataset.transform = testloader.dataset.transform
            train_fsl_loader.dataset.transform = testloader.dataset.transform
            model.module.mode = 'avg_cos'
            model.eval()
            
            model.module.update_fc(trainloader, np.unique(trainset.targets), session)
            vl, va = self.test(model, testloader,train_fsl_loader, self.args, session)
            
        return vl, va

    def base_train(self, model, train_fsl_loader,train_gfsl_loader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        for _, batch in enumerate(zip(train_fsl_loader, train_gfsl_loader)):
            
            support_data, support_label = batch[0][0].cuda(), batch[0][1].cuda()
            query_data, query_label = batch[1][0].cuda(), batch[1][1].cuda()
            model.module.mode = 'classifier'
            logits = model(support_data, query_data, support_label,epoch)
            logits=logits[:,:args.base_class]
            total_loss = F.cross_entropy(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1))
            acc = count_acc(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1))

            lrc = scheduler.get_last_lr()[0]
            #tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_item=total_loss.item()
            
            del logits, total_loss  
        print('Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss_item, acc))
        print('Self.current_way:', model.module.current_way)
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    
    
    def test(self, model, testloader, train_fsl_loader,args, session,validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()

        lgt=torch.tensor([])
        lbs=torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                logits = model.module.forward_many(query)
                logits = logits[:, :test_class]
                
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            va = va.item()

            lgt=lgt.view(-1,test_class)
            lbs=lbs.view(-1)

            
            if validation is not True:
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')

                cm=confmatrix(lgt,lbs,save_model_dir)

                perclassacc=cm.diagonal()
                seenac=np.mean(perclassacc[:args.base_class])
                unseenac=np.mean(perclassacc[args.base_class:])
                print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
                self.result_list.append('Seen Acc:%.5f, Unseen ACC:%.5f' % (seenac,unseenac))
                
                #self.analyze_logits(lgt,lbs,args,session)
        return vl, va

   

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        
        self.args.save_path = self.args.save_path + '%dSC-%dEpo-%.2fT-%dSshot' % (
            self.args.sample_class, self.args.epochs_base, self.args.temperature, self.args.sample_shot)
        
        self.args.save_path = self.args.save_path + '%.5fDec-%.2fMom-%dQ_' % (
            self.args.decay, self.args.momentum, self.args.batch_size_base,)
        

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-MS_%s-Gam_%.2f' % (
                self.args.lr_base, self.args.lrg, mile_stone, self.args.gamma,)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Lr1_%.6f-Lrg_%.5f-Step_%d-Gam_%.2f' % (
                self.args.lr_base, self.args.lrg, self.args.step, self.args.gamma,)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        # if self.args.debug:
        #     self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
