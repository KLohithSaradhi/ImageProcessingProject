import torch
import torch.nn as nn
import utils


class MASK(nn.Module):

    def __init__(self):

        super(MASK, self).__init__()

        # I0
        self.conv_I0_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I0_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc_I0_1 = utils.EncodingBlock(64)
        self.dec_I0_1 = utils.DecodingBlock(84)
        self.conv_I0_3 = nn.Conv2d(85, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I0_4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # I1

        self.conv_I1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc_I1_1 = utils.EncodingBlock(64)
        self.enc_I1_2 = utils.EncodingBlock(64)
        self.dec_I1_1 = utils.DecodingBlock(80)
        self.conv_I1_3 = nn.Conv2d(84, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I1_4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # I2

        self.conv_I2_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc_I2_1 = utils.EncodingBlock(64)
        self.enc_I2_2 = utils.EncodingBlock(64)
        self.enc_I2_3 = utils.EncodingBlock(64)
        self.dec_I2_1 = utils.DecodingBlock(64)
        self.conv_I2_3 = nn.Conv2d(80, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I2_4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        # I3

        self.conv_I3_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I3_3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.selec_I3_1 = utils.SelectiveResidualBlock(64)
        self.conv_I3_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_I3_5 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, I0, I1, I2, I3):

        # ENCODER

        ## I0

        I0 = self.conv_I0_1(I0)
        I0 = self.conv_I0_2(I0)
        enc_I0_1 = self.enc_I0_1(I0)

        ## I1

        I1 = self.conv_I1_1(I1)
        I1 = self.conv_I1_2(I1)
        enc_I1_1 = self.enc_I1_1(I1)
        enc_I1_2 = self.enc_I1_2(enc_I0_1)
        element_sum_I1 = enc_I0_1 + I1

        ## I2

        I2 = self.conv_I2_1(I2)
        I2 = self.conv_I2_2(I2)
        enc_I2_1 = self.enc_I2_1(I2)
        enc_I2_2 = self.enc_I2_2(enc_I1_1)
        enc_I2_3 = self.enc_I2_3(enc_I1_2)
        element_sum_I2_1 = I2 + enc_I1_2 + enc_I1_1

        ## I3

        I3 = self.conv_I3_1(I3)
        I3 = self.conv_I3_2(I3)
        concat_I3_1 = torch.cat((I3, enc_I2_1), dim = 1)
        concat_I3_2 = torch.cat((concat_I3_1, enc_I2_2), dim = 1)
        concat_I3_3 = torch.cat((concat_I3_2, enc_I2_3), dim = 1)
        I3 = self.conv_I3_3(concat_I3_3)
        selRes_I3_1 = self.selec_I3_1(I3)

        # DECODER

        ## I3

        O3 = self.conv_I3_4(selRes_I3_1)
        O3 = self.conv_I3_5(O3)

        ## I2
        decode_I2_1 = self.dec_I2_1(selRes_I3_1, element_sum_I2_1)
        O2 = self.conv_I2_3(decode_I2_1)
        O2 = self.conv_I2_4(O2)

        

        ## I1
        decode_I1_1 = self.dec_I1_1(decode_I2_1, element_sum_I1)
        O1 = self.conv_I1_3(decode_I1_1)
        O1 = self.conv_I1_4(O1)

        ## I0
        decode_I0_1 = self.dec_I0_1(decode_I1_1, I0)
        O0 = self.conv_I0_3(decode_I0_1)
        O0 = self.conv_I0_4(O0)

        return O0, O1, O2, O3

