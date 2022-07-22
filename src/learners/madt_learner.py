import copy
import torch

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.distributions import Categorical

class MADTLearner:
    def __init__(self, mac, args, config, logger):
        self.model = mac.actor
        self.critic_model = mac.critic
        self.args = args
        self.mac = mac
        self.logger = logger
        self.config = config

        # take over whatever gpus are on the system
        self.device = args.device
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.madt.configure_optimizers(config, config.learning_rate)

        self.raw_critic_model = self.critic_model.module if hasattr(self.critic_model, "module") else self.critic_model
        self.critic_optimizer = self.raw_critic_model.madt.configure_optimizers(config, config.learning_rate * 10)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, dataset, t_env, train_critic = True):
        model, critic_model, config = self.raw_model, self.raw_critic_model, self.config
        target_model = copy.deepcopy(model)
        target_model.train(False)

        def run_epoch():
            model.train(True)
            critic_model.train(True)
            if self.config.mode == "offline":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)
            elif self.config.mode == "online":
                loader = DataLoader(dataset, shuffle=True, pin_memory=True, drop_last=True,
                                    batch_size=dataset.__len__(),
                                    num_workers=config.num_workers)
            else:
                raise NotImplementedError

            loss_info = 0
            pbar = tqdm(enumerate(loader), total=len(loader))

            # todo: check these inputs
            for it, (s, o, a, r, ava, v, rtg, ret, adv, t, pre_a, next_s, next_rtg, done) in pbar:
                # place data on the correct device
                s = s.to(self.device)
                o = o.to(self.device)
                a = a.to(self.device)
                r = r.to(self.device)
                ava = ava.to(self.device)
                v = v.to(self.device)
                rtg = rtg.to(self.device)
                ret = ret.to(self.device)
                adv = adv.to(self.device)
                t = t.to(self.device)
                pre_a = pre_a.to(self.device)
                next_s = next_s.to(self.device)
                next_rtg = next_rtg.to(self.device)
                done = done.to(self.device)

                # print("s: {}".format(s.size()))
                # print("o: {}".format(o.size()))
                # print("a: {}".format(a.size()))
                # print("r: {}".format(r.size()))
                # print("pre_a: {}".format(pre_a.size()))
                # print("t: {}".format(t.size()))

                # update actor
                with torch.set_grad_enabled(True):
                    logits = model(o, pre_a, rtg, t)
                    if self.config.mode == "offline":
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a.reshape(-1))
                        entropy_info = 0.
                        ratio_info = 0.
                        confidence_info = 0.
                    elif self.config.mode == "online":
                        adv = adv.reshape(-1, adv.size(-1))

                        logits[ava == 0] = -1e10
                        distri = Categorical(logits=logits.reshape(-1, logits.size(-1)))
                        target_a = a.reshape(-1)
                        log_a = distri.log_prob(target_a).unsqueeze(-1)

                        old_logits = target_model(o, pre_a, rtg, t).detach()
                        old_logits[ava == 0] = -1e10
                        old_distri = Categorical(logits=old_logits.reshape(-1, old_logits.size(-1)))
                        old_log_a = old_distri.log_prob(target_a).unsqueeze(-1)

                        imp_weights = torch.exp(log_a - old_log_a)
                        actor_loss_ori = imp_weights * adv
                        actor_loss_clip = torch.clamp(imp_weights, 1.0 - 0.2, 1.0 + 0.2) * adv
                        actor_loss = -torch.min(actor_loss_ori, actor_loss_clip)
                        # actor_loss = -log_a * adv

                        act_entropy = distri.entropy().unsqueeze(-1)
                        loss = actor_loss - 0.01 * act_entropy
                        # loss = actor_loss

                        entropy_info = act_entropy.mean().item()
                        ratio_info = imp_weights.mean().item()
                        confidence_info = torch.exp(log_a).mean().item()
                    else:
                        raise NotImplementedError
                    loss = loss.mean()
                    loss_info = loss.item()

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # update critic
                critic_loss_info = 0.
                if train_critic:
                    with torch.set_grad_enabled(True):
                        v_value = critic_model(s, pre_a, rtg, t)
                        v_clip = v + (v_value - v).clamp(-0.2, 0.2)
                        critic_loss_ori = F.smooth_l1_loss(v_value.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss_clip = F.smooth_l1_loss(v_clip.view(-1, 1), ret.view(-1, 1), beta=10)
                        critic_loss = torch.max(critic_loss_ori, critic_loss_clip)

                        critic_loss_info = critic_loss.mean().item()

                    critic_model.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_model.parameters(), config.grad_norm_clip)
                    self.critic_optimizer.step()

                # report progress
                pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}.")
            return loss_info, critic_loss_info, entropy_info, ratio_info, confidence_info

        actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = 0., 0., 0., 0., 0.

        for epoch in range(config.max_epochs):
            actor_loss_ret, critic_loss_ret, entropy, ratio, confidence = run_epoch()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("actor_loss", actor_loss_ret, t_env)
            self.logger.log_stat("critic_loss", critic_loss_ret, t_env)
            self.logger.log_stat("entropy", entropy, t_env)
            self.logger.log_stat("confidence", confidence, t_env)
            self.log_stats_t = t_env
        
        return actor_loss_ret, critic_loss_ret, entropy, ratio, confidence

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
