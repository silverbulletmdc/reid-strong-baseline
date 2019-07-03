#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 5, 20

#%%
with open('../output.pkl', 'rb') as f:
    res = pickle.load(f)

print(res.keys())
#%%
print(res['gallery_paths'])
print(res['query_paths'])
#%%
print(len(res['query_paths']))
#%%
#%%
imgs = []
for path in res['query_paths'][:50]:
    img = plt.imread('../' + path)
    imgs.append(img)
plt.figure(figsize=(20,10))
ax = plt.subplot(5, 10, 1)
ax.axis('off')
ax.imshow(imgs[0])
for i in range(1, 50):
    ax = plt.subplot(5, 10, i+1)
    ax.axis('off')
    ax.imshow(imgs[i])
    ax.set_title('%d' % (i), color='green')
#%%
def visulize_one_query(q_index):
    plt.clf()
    topk = 100
    columns = 10
    rows = topk // columns

    q_id = res['query_ids'][q_index]
    q_cam = res['query_cams'][q_index]
    dist_row = res['distmat'][q_index]
    indexs = np.argsort(dist_row)

    q_img = plt.imread('../' + res['query_paths'][q_index])
    ax = plt.subplot(rows, columns, 1)
    ax.axis('off')
    ax.imshow(q_img)

    pos = 2
    for i, index in enumerate(indexs):
        g_cam = res['gallery_cams'][index]
        g_id = res['gallery_ids'][index]
        if g_cam != q_cam:
            color = 'red'
            if g_id == q_id:
                color = 'green'

            q_img = plt.imread('../' + res['gallery_paths'][index])
            ax = plt.subplot(rows, columns, pos)
            ax.axis('off')
            ax.imshow(q_img)
            ax.set_title('%d' % (pos-1), color=color)
            if pos == topk:
                break
            pos += 1

    plt.savefig('imgs/{}.png'.format(q_index))

plt.figure(figsize=(60, 30))
#%%
for i in range(447, len(res['query_paths'])):
    visulize_one_query(i)
    print(i)
#%%

