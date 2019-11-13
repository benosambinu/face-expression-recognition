#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *


# In[3]:


bs = 32


# In[4]:


path = untar_data('/Users/Beno Sam Binu/.fastai/data/face-expression-recognition')


# In[5]:


path


# In[6]:


path.ls()


# In[7]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=200, bs = bs, num_workers=4).normalize(imagenet_stats)


# In[8]:


data.classes


# In[9]:


data.show_batch(rows = 3, figsize = (7,8))


# In[10]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[11]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[12]:


learn.fit_one_cycle(4)


# In[13]:


learn.save('stage-1')


# In[14]:


learn.unfreeze()


# In[15]:


learn.lr_find()


# In[16]:


learn.recorder.plot()


# In[18]:


learn.fit_one_cycle(4,max_lr=slice(1e-5,1e-3))


# In[19]:


learn.save('stage-2')


# In[20]:


learn.load('stage-2')


# In[21]:


interp = ClassificationInterpretation.from_learner(learn)


# In[22]:


interp.plot_confusion_matrix()


# In[23]:


learn.export()


# In[24]:


defaults.device = torch.device('cpu')


# In[28]:


img = open_image(path/'valid'/'sad'/'350.jpg')
img


# In[29]:


learn = load_learner(path)


# In[30]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:




