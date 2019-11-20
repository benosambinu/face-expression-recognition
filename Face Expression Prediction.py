#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *

bs = 32
path = untar_data('/Users/Beno Sam Binu/.fastai/data/face-expression-recognition')
path.ls()
np.random.seed(42)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=200, bs = bs, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows = 3, figsize = (7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-3))
learn.save('stage-2')
learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
learn.export()

defaults.device = torch.device('cpu')
img = open_image(path/'valid'/'sad'/'350.jpg')
img

learn = load_learner(path)

pred_class,pred_idx,outputs = learn.predict(img)
pred_class




