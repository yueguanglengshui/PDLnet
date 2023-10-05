import matplotlib.pyplot as plt

loss_img_save = './accuracy_loss.jpg'


def draw_loss_and_accuracy(Loss_list, Accuracy_list, train_epoch):
    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig(loss_img_save)
