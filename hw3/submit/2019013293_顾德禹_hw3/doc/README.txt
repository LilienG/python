2.3 绘制Loss曲线
第三方库：matplotlib，pylab

2.4 更换优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
SGD效果不如Adam

2.5 添加数据预处理
添加了RandomCrop,RandomHorizontalFlip,RandomVerticalFlip
因为对于原数据增加了随机水平&垂直翻转以及随机裁剪，这时的训练的难度增加，但可以看到经多次训练准确度仍能从一开始的63%调高到90%左右，泛化能力增强。

2.6 增加一层卷积
可以看到准确率得到提升，尤其是在训练条件苛刻的情况下。
然后使用Netron工具对训练得到的lenet.pth进行建模输出。

2.7 关于错误数据输出
打印出的手写体图片都比较抽象，对于机器学习存在较大的犯错概率。具体来说，可能引起错误的地方有：笔画过短或过长，笔画末端有收笔连带，整体与笔画间均存在一定倾斜。这些都会导致学习到的参数不适用，进行重复训练调整。