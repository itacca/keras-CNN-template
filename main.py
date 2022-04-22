# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    # print_hi('PyCharm')

    # Run the following command before using GPU:
    # sudo ldconfig /usr/lib/cuda/lib64
    import tensorflow as tf
    print(tf.config.list_physical_devices("GPU"))

    # with tf.device('/gpu:0'):
    #     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    #     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    #     c = tf.matmul(a, b)
    # tf.print(c)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
