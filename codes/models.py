import torch
import torch.nn.functional as F
import torch.nn as nn
import octconv as oc
class MS_RB(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1 + x2
        x4 = self.conv4(x3)
        out = x4 + x

        return out

def resample_data(input, s):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """    
    
    assert( not input.size(2)%s and not input.size(3)%s)
    
    if input.size(1) == 3:
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:,2] + 0.587 * input[:,1] + 0.114 * input[:,0]).unsqueeze(1)
        
    out = torch.cat([input[:,:,i::s,j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out

class Net(nn.Module):
    def __init__(self, num_feats, depth_chanels, color_channel, kernel_size):
        super(Net, self).__init__()
        self.conv_rgb1 = nn.Conv2d(in_channels=16, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        #self.conv_rgb2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb3 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        #self.conv_rgb5 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
                                   
        self.rgb_cbl2 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rgb_cbl3 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))   
        self.rgb_cbl4 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #self.rgb_cbl5 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.125, alpha_out=0.125,
        #                            stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))                         
                                   
                                   
    


        self.conv_dp1 = nn.Conv2d(in_channels=16, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.MSB1 = MS_RB(num_feats, kernel_size)
        self.MSB2 = MS_RB(56, kernel_size)
        self.MSB3 = MS_RB(80, kernel_size)
        self.MSB4 = MS_RB(104, kernel_size)

        self.conv_recon1 = nn.Conv2d(in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.conv_recon2=nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.restore=nn.Conv2d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        image, depth = x

        re_im = resample_data(image, 4)
        re_dp = resample_data(depth, 4)
        
        dp_in = self.act(self.conv_dp1(re_dp))
        dp1 = self.MSB1(dp_in)

        rgb1 = self.act(self.conv_rgb1(re_im))
        #rgb2 = self.act(self.conv_rgb2(rgb1))
        
        rgb2 = self.rgb_cbl2(rgb1)
        
        ca1_in = torch.cat([dp1,rgb2[0]],dim = 1)
        dp2 = self.MSB2(ca1_in)
        #rgb3 = self.conv_rgb3(rgb2)
        rgb3 = self.rgb_cbl3(rgb2)
        #ca2_in = dp2 + rgb3
        ca2_in = torch.cat([dp2,rgb3[0]],dim = 1)

        dp3 = self.MSB3(ca2_in)
        #rgb4 = self.conv_rgb4(rgb3)
        rgb4 = self.rgb_cbl4(rgb3)

        #ca3_in = rgb4 + dp3
        ca3_in = torch.cat([dp3,rgb4[0]],dim = 1)
        
        dp4 = self.MSB4(ca3_in)
        up1 = self.ps1(self.conv_recon1(self.act(dp4)))
        up2 = self.ps2(self.conv_recon2(up1))
        out = self.restore(up2)
        out = depth + out

        return out