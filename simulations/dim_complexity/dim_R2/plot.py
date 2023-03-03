import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # layer 3
    # Read data
    plt.clf()
    df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer1_v1/results.csv')
    df4 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer1_v2/results.csv')
    df5 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer1_v3/results.csv')
    df6 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=nsd_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer1_v4/results.csv')

    # gca stands for 'get current axis'
    df3['low_dim'] = np.log10(df3['low_dim'])
    df4['low_dim'] = np.log10(df4['low_dim'])
    df5['low_dim'] = np.log10(df5['low_dim'])
    df6['low_dim'] = np.log10(df6['low_dim'])
    ax = plt.gca()

    df3.plot(kind='line',x='low_dim',y='test_r2',ax=ax, label='Res18Pre')
    df4.plot(kind='line',x='low_dim',y='test_r2', color='red', ax=ax, label='Res18Pre')
    df5.plot(kind='line',x='low_dim',y='test_r2', color='green', ax=ax, label='Res18Pre')
    df6.plot(kind='line',x='low_dim',y='test_r2', color='yellow', ax=ax, label='Res18Pre')

    plt.xlabel('Number of dimensions')
    plt.ylabel('R^2')
    plt.title('Resnet18 layer1 Nsd Multiple Runs')
    plt.savefig('/home/swadhwa5/projects/encoder_warping/simulations/dim_complexity/saved_figures/test/dim_R2_plot_layer1_multiple_runs.png')

if __name__ == "__main__":
    main()