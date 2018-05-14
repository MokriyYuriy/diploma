import torch.nn as nn
import torch.nn.functional as F

from ..loss import policy_loss, disc_loss, cross_entropy


class GAN(nn.Module):
    def __init__(self, gen_model, disc_model):
        super(GAN, self).__init__()
        self.gen_model = gen_model
        self.disc_model = disc_model

    def baseline_forward(self, input_sequence):
        baseline_sequence = self.gen_model.translate(input_sequence, strategy='greedy', return_logits=True)[1]
        return self.disc_model(baseline_sequence.detach())

    def src_forward(self, input_sequence):
        result_sequence, logits \
            = self.gen_model.translate(input_sequence, strategy='sampling', return_logits=True)
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
        disc_predictions, result_sequence, logits = src_gan.src_forward(input_sequence)
        baseline_disc_predictions = src_gan.baseline_forward(input_sequence)
        reversed_logits = trg_gan.gen_model(result_sequence)
        return disc_predictions, result_sequence, logits, baseline_disc_predictions, reversed_logits

    def disc_forward(self, output_sequence, reversed=False):
        if reversed:
            return self.trg_gan.trg_forward(output_sequence)
        else:
            return self.src_gan.trg_forward(output_sequence)

    def compute_losses(self, input_sequence, reversed=False):
        if reversed:
            src_gan, trg_gan = self.trg_gan, self.src_gan
        else:
            src_gan, trg_gan = self.src_gan, self.trg_gan
        disc_predictions, result_sequence, logits, baseline_disc_predictions, reversed_logits \
            = self.forward(input_sequence, reversed)
        src_alphabet = src_gan.gen_model.encoder.alphabet
        trg_alphabet = trg_gan.gen_model.encoder.alphabet
        pg_loss = policy_loss(disc_predictions, baseline_disc_predictions, logits, trg_alphabet)
        disc_cross_entropy = disc_loss(self.disc_forward(input_sequence), baseline_disc_predictions)
        cycle_loss = cross_entropy(src_gan.trg_gan.gen_model(result_sequence), input_sequence, src_alphabet)
        return pg_loss, disc_cross_entropy, cycle_loss



