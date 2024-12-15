from fastai.collab import *
from fastai.tabular.all import *

path = untar_data(URLs.ML_100k)

# MovieLens. This dataset contains tens of millions of movie rankings (a combination of a movie ID,
# a user ID, and a numeric rat‐ ing), although we will just use a subset of 100,000 of them.


# columns are, respectively, user, movie, rating, and timestamp.
ratings = pd.read_csv(
    path / "u.data",
    delimiter="\t",
    header=None,
    names=["user", "movie", "rating", "timestamp"],
)

ratings.header()

# Here, for instance, we are scoring very science-fiction as 0.98, and very not old as –0.9.
last_skywalker = np.array([0.98, 0.9, -0.9])

# We could represent a user who likes modern sci-fi action movies as follows:
user1 = np.array([0.9, 0.8, -0.6])

# We can now calculate the match between this combination:
(user1 * last_skywalker).sum()  # 2.1420000000000003

# On the other hand, we might represent the movie Casablanca (1942) as follows:
casablanca = np.array([-0.99, -0.3, 0.8])

(user1 * casablanca).sum()  # -1.611


movies = pd.read_csv(
    path / "u.item",
    delimiter="|",
    encoding="latin-1",
    usecols=(0, 1),
    names=("movie", "title"),
    header=None,
)

movies.read()

# We can merge this with our ratings table to get the user ratings by title:
ratings = ratings.merge(movies)
ratings.head()

dls = CollabDataLoaders.from_df(ratings, item_name="title", bs=64)
dls.show_batch()

# We can represent our movie and user latent factor tables as simple matrices:
n_users = len(dls.classes["user"])
n_movies = len(dls.classes["title"])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

# represent look up in an index as a matrix product. The trick is to replace our
# indices with one-hot-encoded vectors

one_hot_3 = one_hot(3, n_users).float()

# E,g of what happens if we multiply a vector by a one-hot-encoded vector representing the index 3:
user_factors.t() @ one_hot_3  # tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])

# It gives us the same vector as the one at index 3 in the matrix:
user_factors[3]  #  tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])


# The final thing that you need to know to create a new PyTorch module is that when your module is
# called, PyTorch will call a method in your class called forward, and will pass along to that any
# parameters that are included in the call. Here is the class defining our dot product model:


class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range

    # Note that the input of the model is a tensor of shape batch_size x 2, where the first
    # column (x[:, 0]) contains the user IDs, and the second column (x[:, 1]) contains the movie IDs.
    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        # The first thing we can do to make this model a little bit better is to force
        # those predictions to be between 0 and 5. For this, we just need to use sigmoid_range,
        return sigmoid_range((users * movies).sum(dim=1), *self.y_range)


# we use the embedding layers to represent our matrices of user and movie latent factors:
x, y = dls.one_batch()
x.shape  # torch.Size([64, 2])

model = DotProduct(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())

# We are now ready to fit our model:
learn.fit_one_cycle(5, 5e-3)


class DotProductBias(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)
# Now, Instead of being better, it ends up being worse

# weight decay (or just wd) is a parameter that controls that sum of squares we
# add to our loss (assuming parameters is a tensor of all parameters):
# loss_with_wd = loss + wd * (parameters**2).sum()

model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


class T(Module):
    def __init__(self):
        self.a = torch.ones(3)


print(L(T().parameters()))  # (#0) []


class T2(Module):
    # nn.Parameters: To tell Module that we want to treat a tensor as a parameter,
    def __init__(self):
        self.a = nn.Parameter(torch.ones(3))


print(L(T().parameters()))
# (#1) [Parameter containing: tensor([1., 1., 1.], requires_grad=True)]


class T3(Module):
    def __init__(self):
        self.a = nn.Linear(1, 3, bias=False)


t = T3()
L(t.parameters())
# (#1) [Parameter containing: tensor([[-0.9595], [-0.8490], [ 0.8159]], requires_grad=True)]

type(t.a.weight)  # torch.nn.parameter.Parameter


# We can create a tensor as a parameter, with random initialization, like so:
def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


# Let’s use this to create DotProductBias again, but without Embedding:
class DotProductBias2(Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors[x[:, 0]]
        movies = self.movie_factors[x[:, 1]]
        res = (users * movies).sum(dim=1)
        res += self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
        return sigmoid_range(res, *self.y_range)


model = DotProductBias2(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


# Interpreting Embeddings and Biases
# Here are the movies with the lowest values in the bias vector:
movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes["title"][i] for i in idxs]

# By the same token, here are the movies with the highest bias:
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes["title"][i] for i in idxs]


# create and train a collaborative filtering model using the exact structure shown earlier:
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.fit_one_cycle(5, 5e-3, wd=0.1)

# The names of the layers can be seen by printing the model:
learn.model
# EmbeddingDotBias(
# (u_weight): Embedding(944, 50)
# (i_weight): Embedding(1635, 50)
# (u_bias): Embedding(944, 1)
# (i_bias): Embedding(1635, 1)
# )

# We can use these to replicate any of the analyses we did in the previous section:
movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes["title"][i] for i in idxs]


# Embedding Distance
# If there were two movies that were nearly identical, their
# embedding vectors would also have to be nearly identical,
movie_factors = learn.model.i_weight.weight
idx = dls.classes["title"].o2i["Silence of the Lambs, The (1991)"]
distance = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distance.argsort(descending=True)[1]
dls.classes["title"][idx]  # 'Dial M for Murder (1954)'


#
embs = get_emb_sz(dls)  # returns recommended sizes for embedding matrices
embs  # [(944, 74), (1635, 101)]


# Let’s implement this class:
class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0, 5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1] + item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1),
        )
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.item_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)


# And use it to create a model:
model = CollabNN(*embs)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)

learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100, 50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)
