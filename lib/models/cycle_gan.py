import torch.nn as nn
import torch.nn.functional as F

from ..loss import policy_loss, disc_cross_entropy, cross_entropy


class GAN(nn.Module):
    def __init__(self, gen_model, disc_model):
        super(GAN, self).__init__()
        self.gen_model = gen_model
        self.disc_model = disc_model

    def src_forward(self, input_sequence, strategy):
        result_sequence, logits \
            = self.gen_model.translate(input_sequence, strategy=strategy, return_logits=True)
        disc_predictions = self.disc_model(result_sequence.detach())
        return disc_predictions, result_sequence, logits

    def trg_forward(self, output_sequence):
        return self.disc_model(output_sequence.detach())


class CycleGAN(nn.Module):
    def __init__(self, src_gan, trg_gan):
        super(CycleGAN, self).__init__()
        self.src_gan = src_gan
        self.trg_gan = trg_gan

    def forward(self, input_sequence, reversed=False):
        if reversed:
            src_gan, trg_gan = self.trg_gan, self.src_gan
        else:
            src_gan, trg_gan = self.src_gan, self.trg_gan
        disc_predictions, result_sequence, logits = src_gan.src_forward(input_sequence, strategy='sampling')
        baseline_disc_predictions, baseline_result_sequence, _ \
            = src_gan.src_forward(input_sequence, strategy='greedy')

        reversed_logits = trg_gan.gen_model(result_sequence, input_sequence[:, 1:].contiguous())
        return disc_predictions, result_sequence, logits, \
               baseline_disc_predictions, baseline_result_sequence, reversed_logits

    def disc_forward(self, output_sequence, reversed=False):
        if reversed:
            return self.src_gan.trg_forward(output_sequence)
        else:
            return self.trg_gan.trg_forward(output_sequence)

    def compute_losses(self, input_sequence, reversed=False):
        if reversed:
            src_gan, trg_gan = self.trg_gan, self.src_gan
        else:
            src_gan, trg_gan = self.src_gan, self.trg_gan

        (
            disc_predictions,
            result_sequence,
            logits,
            baseline_disc_predictions,
            baseline_result_sequence,
            reversed_logits
        ) = self.forward(input_sequence, reversed)

        src_alphabet = src_gan.gen_model.encoder.alphabet
        trg_alphabet = trg_gan.gen_model.encoder.alphabet

        rev_true_disc_loss, fwd_fake_disc_loss = disc_cross_entropy(
            self.disc_forward(input_sequence, reversed), disc_predictions, alpha=src_gan.disc_model.alpha, sep_return=True
        )

        cycle_cross_entropy = cross_entropy(
            trg_gan.gen_model(result_sequence.detach(), input_sequence),
            input_sequence[:, 1:].contiguous(),
            src_alphabet,
            reduce_mean=False
        )

        baseline_cycle_cross_entropy = cross_entropy(
            trg_gan.gen_model(baseline_result_sequence.detach(), input_sequence),
            input_sequence[:, 1:].contiguous(),
            src_alphabet,
            reduce_mean=False
        )

        forward_advantages = F.logsigmoid(disc_predictions) - F.logsigmoid(baseline_disc_predictions)
        cycle_advantages = baseline_cycle_cross_entropy - cycle_cross_entropy

        logits = logits.contiguous()
        mask = trg_alphabet.get_mask_for_3D_array(result_sequence[:, 1:].contiguous(), logits).contiguous()
        normalized_logits = F.log_softmax(logits, dim=2)
        pg_discr_loss = policy_loss(forward_advantages, normalized_logits, mask)
        pg_cycle_loss = policy_loss(cycle_advantages, normalized_logits, mask)
        entropy = -(F.softmax(logits, dim=2) * normalized_logits * mask).sum(1).sum(1)
        pg_entropy = policy_loss(-entropy, normalized_logits, mask)

        advantages = dict(
            disc_advantage=forward_advantages.mean().data[0],
            cycle_advantage=cycle_advantages.mean().data[0],
            entropy=entropy.mean().data[0],
            disc_reward=F.logsigmoid(disc_predictions).mean().data[0]
        )

        return pg_discr_loss, pg_cycle_loss, rev_true_disc_loss, fwd_fake_disc_loss, cycle_cross_entropy.mean(), pg_entropy, advantages

