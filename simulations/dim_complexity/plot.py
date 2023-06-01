import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # layer 3
    # Read data
    plt.clf()
    df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/test.csv')
    # df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=convnext_dataset=imagenet_task=object-classification_layer=avgpool/results.csv')
    # df4 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=vit16_dataset=imagenet_task=object-classification_layer=heads/results.csv')
    
    # gca stands for 'get current axis'
    df3['low_dim'] = np.log10(df3['low_dim'])
    # df4['low_dim'] = np.log10(df4['low_dim'])
    ax = plt.gca()
    
    # ax.axhline(y=0.09, xmin=0, xmax=3, c="black", linestyle = '--', linewidth=1, zorder=0)

    df3.plot(kind='line',x='low_dim',y='test_r2',ax=ax, label='Enginnered Model')
    # df3.plot(kind='line',x='low_dim',y='test_r2',ax=ax, label='ConvNext')
    # df4.plot(kind='line',x='low_dim',y='test_r2', color='red', ax=ax, label='VIT16')

    plt.xlabel('Number of dimensions')
    plt.ylabel('R^2')
    plt.title('Enginnered Model')
    # plt.title('ConvNext vs VIT16 on NSD')
    plt.savefig('/home/swadhwa5/projects/encoder_warping/simulations/dim_complexity/saved_figures/nsd/engineered_model.png')

if __name__ == "__main__":
    main()