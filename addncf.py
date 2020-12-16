import math
import copy
import pickle
import zipfile
from itertools import zip_longest
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# For neural network
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F 
from torch.optim.lr_scheduler import _LRScheduler

# Set random seed
def set_random_seed(state=650):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 650
set_random_seed(RANDOM_STATE)

def convert_review(pred):
    pred_review = []
    for i in pred:
        if i <= 1.5:
            pred_review.append(1.0)
        elif i > 1.5 and i <= 2.5:
            pred_review.append(2.0)
        elif i > 2.5 and i <= 3.5:
            pred_review.append(3.0)
        elif i > 3.5 and i <= 4.5:
            pred_review.append(4.0)
        else:
            pred_review.append(5.0)
    return pred_review

class ReviewsIterator:
    
    def __init__(self, X, y, batch_size=16, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)
        
        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]
            
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k*bs:(k + 1)*bs], self.y[k*bs:(k + 1)*bs]

def batches(X, y, bs=16, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)

df_sample = pd.read_csv('sephora_review_skincare_sample.csv')
# Create user table and item table
df_user = df_sample[['user_id', 'product_id', 'rating', 
                     'skin_type', 'skin_tone', 'skin_concerns']].reset_index(drop=True)
df_user.skin_type.fillna('no_answer', inplace=True)
df_user.skin_tone.fillna('no_answer', inplace=True)
df_user.skin_concerns.fillna('no_answer', inplace=True)

df_item = df_sample[['product_id', 'brand_id', 'description', 'price']].reset_index(drop=True)

def pick_lowprice(price):
    idx = price.find('-')
    if idx == -1:
        return float(price[1:])
    else:
        return float(price[1:idx - 1])

def pick_highprice(price):
    idx = price.find('-')
    if idx == -1:
        return float(price[1:])
    else:
        return float(price[idx + 3:])

df_item['low_price'] = df_item.price.map(pick_lowprice)
df_item['high_price'] = df_item.price.map(pick_highprice)
df_item['rprice'] = 0.5 * (df_item.high_price + df_item.low_price)
df_item.drop(['price', 'low_price', 'high_price'], axis=1, inplace=True)

df_item['len_des'] = df_item.description.map(len)
df_item.drop_duplicates(['product_id','rprice'], inplace=True)
df_item.reset_index(drop=True, inplace=True)
df_item.drop('len_des', axis=1, inplace=True)

def map_price(price):
    if price <= 26.00:
        return 0
    elif price > 26.00 and price <= 40.00:
        return 1
    elif price > 40.00 and price <= 60.875:
        return 2
    else:
        return 3

df_item['price_band'] = df_item.rprice.map(map_price)
df_item.drop('rprice', axis=1, inplace=True)

df_user_item = df_user.merge(df_item, how='left', left_on='product_id', right_on='product_id')

# Define dataset and functions
def create_dataset_addfeature(ratings):
    
    unique_users = ratings.user_id.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.user_id.map(user_to_index)
    
    unique_products = ratings.product_id.unique()
    product_to_index = {old: new for new, old in enumerate(unique_products)}
    new_products = ratings.product_id.map(product_to_index)

    unique_skin_types = ratings.skin_type.unique()
    skintype_to_index = {old: new for new, old in enumerate(unique_skin_types)}
    new_skintypes = ratings.skin_type.map(skintype_to_index)

    unique_skin_tones = ratings.skin_tone.unique()
    skintone_to_index = {old: new for new, old in enumerate(unique_skin_tones)}
    new_skintones = ratings.skin_tone.map(skintone_to_index)

    unique_concerns = ratings.skin_concerns.unique()
    concern_to_index = {old: new for new, old in enumerate(unique_concerns)}
    new_concerns = ratings.skin_concerns.map(concern_to_index)

    unique_brands = ratings.brand_id.unique()
    brand_to_index = {old: new for new, old in enumerate(unique_brands)}
    new_brands = ratings.brand_id.map(brand_to_index)

    n_users = unique_users.shape[0]
    n_products = unique_products.shape[0]
    n_price = len(ratings.price_band.unique())
    n_skintypes = unique_skin_types.shape[0]
    n_skintones = unique_skin_types.shape[0]
    n_concerns = unique_concerns.shape[0]
    n_brands = unique_brands.shape[0]
    
    X = pd.DataFrame({'user_id': new_users, 'product_id': new_products,
                      'price_id': ratings.price_band, 'skintype_id': new_skintypes, 
                      'skintone_id': new_skintypes,'concerns_id': new_concerns,
                      'brand_id': new_brands})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_products, n_price, n_skintypes, n_skintones, 
            n_concerns, n_brands), (X, y), (user_to_index, product_to_index, 
                                            skintype_to_index, skintone_to_index,
                                            concern_to_index, brand_to_index)

# Define network
class AddFeaturesEmbeddingNet(nn.Module):
    """
    Creates a dense network with embedding layers.
    
    Args:
        n_users:            
            Number of unique users in the dataset.
        n_products: 
            Number of unique products in the dataset.
        n_factors: 
            Number of columns in the embeddings matrix.
        embedding_dropout: 
            Dropout rate to apply right after embeddings layer.
        hidden:
            A single integer or a list of integers defining the number of 
            units in hidden layer(s).
        dropouts: 
            A single integer or a list of integers defining the dropout 
            layers rates applyied right after each of hidden layers.       
    """
    def __init__(self, n_users, n_products, n_prices,
                 n_skintypes, n_skintones, n_concerns,
                 n_brands, n_factors=50, n_factors_fix=30,
                 embedding_dropout=0.02, 
                 hidden=10, dropouts=0.2):
        
        super().__init__()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        n_last = hidden[-1]
        
        def gen_layers(n_in):
            """
            A generator that yields a sequence of hidden layers and 
            their activations/dropouts.
            
            Note that the function captures `hidden` and `dropouts` 
            values from the outer scope.
            """
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)
            
            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out
            
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.product_embedding = nn.Embedding(n_products, n_factors)
        self.price_embedding = nn.Embedding(n_prices, n_factors_fix)
        self.skintype_embedding = nn.Embedding(n_skintypes, n_factors_fix)
        self.skintone_embedding = nn.Embedding(n_skintones, n_factors_fix)
        self.concern_embedding = nn.Embedding(n_concerns, n_factors_fix)
        self.brand_embedding = nn.Embedding(n_brands, n_factors_fix)

        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 2 + n_factors_fix * 5)))
        self.fc = nn.Linear(n_last, 1)
        self._init()
        
    def forward(self, users, products, prices, 
                skintypes, skintones, concerns, brands,
                minmax=None):
       
        features = torch.cat([self.user_embedding(users), 
                              self.product_embedding(products),
                              self.price_embedding(prices),
                              self.skintype_embedding(skintypes),
                              self.skintone_embedding(skintones),
                              self.concern_embedding(concerns),
                              self.brand_embedding(brands)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out*(max_rating - min_rating + 1) + min_rating - 0.5
        return out
    
    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """
        
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.user_embedding.weight.data.uniform_(-0.1, 0.1)
        self.product_embedding.weight.data.uniform_(-0.1, 0.1)
        self.price_embedding.weight.data.uniform_(-0.1, 0.1)
        self.skintype_embedding.weight.data.uniform_(-0.1, 0.1)
        self.skintone_embedding.weight.data.uniform_(-0.1, 0.1)
        self.concern_embedding.weight.data.uniform_(-0.1, 0.1)
        self.brand_embedding.weight.data.uniform_(-0.1, 0.1)
        self.hidden.apply(init)
        init(self.fc)
    
    
def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuraiton should be a single number or a list of numbers')


(n_user, n_product, n_price, n_skintype, 
 n_skintone, n_concern, n_brand), (X, y), (user_to_index, product_to_index, 
                                            skintype_to_index, skintone_to_index,
                                            concern_to_index, brand_to_index) = create_dataset_addfeature(df_user_item)

# Scheduler
class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def triangular(step_size, max_lr, method='triangular', gamma=0.99):
    
    def scheduler(epoch, base_lr):
        period = 2 * step_size
        cycle = math.floor(1 + epoch/period)
        x = abs(epoch/step_size - 2*cycle + 1)
        delta = (max_lr - base_lr)*max(0, (1 - x))

        if method == 'triangular':
            pass  # we've already done
        elif method == 'triangular2':
            delta /= float(2 ** (cycle - 1))
        elif method == 'exp_range':
            delta *= (gamma**epoch)
        else:
            raise ValueError('unexpected method: %s' % method)
            
        return base_lr + delta
        
    return scheduler

def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler

# Data spilit into training data and validation dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

minmax = float(df_user.rating.min()), float(df_user.rating.max())

lr = 1e-3
wd = 1e-5
bs = 64
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
addnet.to(device)
criterion = nn.L1Loss(reduce='sum')
optimizer = optim.Adam(addnet.parameters(), lr=lr, weight_decay=wd)
iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, 
                                       eta_min=lr/10))

for epoch in range(n_epochs):
    stats = {'epoch': epoch + 1, 'total': n_epochs}
    
    for phase in ('train', 'val'):
        training = phase == 'train'
        running_loss = 0.0
        n_batches = 0
        
        for batch in batches(*datasets[phase], shuffle=training, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
        
            # compute gradients only during 'train' phase
            with torch.set_grad_enabled(training):
                outputs = addnet(x_batch[:, 0], x_batch[:, 1],
                                 x_batch[:, 2], x_batch[:, 3],
                                 x_batch[:, 4], x_batch[:, 5],
                                 x_batch[:, 6], minmax)
                loss = criterion(outputs, y_batch)
                
                # don't update weights and rates when in 'val' phase
                if training:
                    scheduler.step()
                    loss.backward()
                    optimizer.step()
                    lr_history.extend(scheduler.get_lr())
                    
            running_loss += loss.item()
            
        epoch_loss = running_loss / dataset_sizes[phase]
        stats[phase] = epoch_loss
        
        # early stopping: save weights of the best model so far
        if phase == 'val':
            if epoch_loss < best_loss:
                print('loss improvement on epoch: %d' % (epoch + 1))
                best_loss = epoch_loss
                best_weights = copy.deepcopy(addnet.state_dict())
                no_improvements = 0
            else:
                no_improvements += 1
                
    history.append(stats)
    print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
    if no_improvements >= patience:
        print('early stopping after epoch {epoch:03d}'.format(**stats))
        break

addnet.load_state_dict(best_weights)

model_path = 'Addemmbednet.pth'
torch.save(addnet.to('cpu').state_dict(), model_path)

model_path = 'Addemmbednet.pth'
addnet.load_state_dict(torch.load(model_path,
                               map_location=torch.device('cpu')))

groud_truth, predictions = [], []
bs = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    for batch in batches(*datasets['val'], shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]
        outputs = addnet(x_batch[:, 0], x_batch[:, 1], 
                         x_batch[:, 2], x_batch[:, 3],
                         x_batch[:, 4], x_batch[:, 5],
                         x_batch[:, 6], minmax)
        groud_truth.extend(y_batch.tolist())
        predictions.extend(outputs.tolist())

ground_truth = np.asarray(groud_truth).ravel()
predictions = np.asarray(predictions).ravel()

final_loss = mean_absolute_error(ground_truth, predictions)
print(f'Final MAE: {final_loss:.4f}')

pred_round = convert_review(predictions)
plt.hist(ground_truth, label='True', align='left')
plt.hist(pred_round, label='NCF')
plt.xlabel('rating', size=12)
plt.ylabel('frequency', size=12)
plt.title('Result of NCF')
plt.grid(True)
plt.legend()