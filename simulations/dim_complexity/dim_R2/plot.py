import pandas as pd
import matplotlib.pyplot as plt

def main():
    # layer 3
    # Read data
    plt.clf()
    df3 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=majaj_arch=resnet18_dataset=imagenet_task=object-classification_layer=layer1/results.csv')
    df4 = pd.read_csv ('/home/swadhwa5/projects/encoder_warping/saved_runs/encoding_dim/neural-data=majaj_arch=resnet18_dataset=None_task=object-classification_layer=layer1/results.csv')
    # gca stands for 'get current axis'
    ax = plt.gca()

    df3.plot(kind='line',x='low_dim',y='test_r2',ax=ax, label='Res18Pre')
    df4.plot(kind='line',x='low_dim',y='test_r2', color='red', ax=ax, label='Res18NoPre')
    plt.xlabel('Number of dimensions')
    plt.ylabel('R^2')
    plt.title('Resnet18 layer3')
    plt.savefig('/home/swadhwa5/projects/encoder_warping/simulations/dim_complexity/saved_figures/majaj/dim_R2_plot_layer1.png')


if __name__ == "__main__":
    main()