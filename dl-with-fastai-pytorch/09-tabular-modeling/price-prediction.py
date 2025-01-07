from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz import *
from IPython.display import Image, display_svg, SVG
from kaggle import *
from sklearn.inspection import plot_partial_dependence

creds = "API key to use the Kaggle API"

cred_path = Path("~/.kaggle/kaggle.json").expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write(creds)
    cred_path.chmod(0o600)

# download datasets from Kaggle!
path = URLs.path("bluebook")
path  # Path('/home/sgugger/.fastai/archive/bluebook')

if not path.exists():
    path.mkdir()
    api.competition_download_cli("bluebook-for-bulldozers", path=path)
    file_extract(path / "bluebook-for-bulldozers.zip")

path.ls(file_type="text")
# (#7) [Path('Valid.csv'),Path('Machine_Appendix.csv'),Path('ValidSolution.csv'),
# Path('TrainAndValid.csv'),Path('random_forest_benchmark_test.csv'),Path('Test.csv'),
# Path('median_benchmark.csv')]

# SalesID: The unique identifier of the sale.
# MachineID: The unique identifier of a machine. A machine can be sold multiple times.
# saleprice: What the machine sold for at auction (provided only in train.csv).
# saledate: The date of the sale.


df = pd.read_csv(path / "TrainAndValid.csv", low_memory=False)
df.columns
# Index(['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'YearMade',
# 'MachineHoursCurrentMeter', 'UsageBand', 'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
# 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup',
# 'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick',
# 'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower',
# 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System',
# 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
# 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls',
# 'Differential_Type', 'Steering_Controls'], dtype='object')

df["ProductSize"].unique()  # handle ordinal columns (strings have a natural ordering)
# array([nan, 'Medium', 'Small', 'Large / Medium', 'Mini', 'Large', 'Compact'], dtype=object)

# We can tell Pandas about a suitable ordering of these levels like so:
sizes = "Large", "Large / Medium", "Medium", "Small", "Mini", "Compact"
df["ProductSize"] = df["ProductSize"].astype("category")
df["ProductSize"].cat.set_categories(sizes, ordered=True, inplace=True)

# we take the log of the prices, so that the m_rmse of that value will give us what we ultimately need:
dep_var = "SalePrice"
df[dep_var] = np.log(df[dep_var])


# we replace every date column with a set of date metadata columns, such as holiday,
# day of week, and month (to make the days context more meaningfull)
df = add_datepart(df, "saledate")

# Let’s do the same for the test set while we’re there:
df_test = pd.read_csv(path / "Test.csv", low_memory=False)
df_test = add_datepart(df, "saledate")

# We can see that there are now lots of new columns in our DataFrame:
" ".join(o for o in df.columns if o.startswith("sale"))
# 'saleYear saleMonth saleWeek saleDay saleDayofweek saleDayofyear saleIs_month_end saleIs_month_start
# saleIs_quarter_end saleIs_quarter_start saleIs_year_end saleIs_year_start saleElapsed'

procs = [Categorify, FillMissing]

cond = (df.saleYear < 2011) | (df.saleMonth < 10)
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx), list(valid_idx))
# TabularPandas needs to be told which columns are continuous and which are catego‐ rical.
cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

# A TabularPandas behaves a lot like a fastai Datasets object, including providing train and valid attrs:
len(to.train), len(to.valid)  # (404710, 7988)

to.show(3)  # We can see that the data is still displayed as strings for categories

# However, the underlying items are all numeric:
to.items.head(3)
to.classes["ProductSize"]  #  We can see the mapping by looking at the classes attr
# (#7) ['#na#','Large','Large / Medium','Medium','Small','Mini','Compact']

(path / "to.pkl").save()

to = (path / "to.pkl").load()  # To read this back later

# Creating the Decision Tree
# To begin, we define our independent and dependent variables:
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

# Now that our data is all numeric, and there are no missing values, create decision tree:
m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y)

# To keep it simple, we’ve told sklearn to create just four leaf nodes.
# To see what it’s learned, we can display the tree:
draw_tree(m, xs, size=7, leaves_parallel=True, precision=2)

# We can show the same information using Terence Parr’s powerful dtreeviz library:
samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(
    m,
    xs.iloc[samp_idx],
    y.iloc[samp_idx],
    xs.columns,
    dep_var,
    fontname="DejaVu Sans",
    scale=1.6,
    label_fontsize=10,
    orientation="LR",
)

xs.loc[xs["YearMade"] < 1900, "YearMade"] = 1950
valid_xs.loc[valid_xs["YearMade"] < 1900, "YearMade"] = 1950

m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)
dtreeviz(
    m,
    xs.iloc[samp_idx],
    y.iloc[samp_idx],
    xs.columns,
    dep_var,
    fontname="DejaVu Sans",
    scale=1.6,
    label_fontsize=10,
    orientation="LR",
)

#  build a bigger tree. Here, we are not passing in any stopping criteria such as max_leaf_nodes:
m = DecisionTreeRegressor()
m.fit(xs, y)


# We’ll create a little function to check the root mean squared error
# of our model (m_rmse), since that’s how the competition was judged:
def r_mse(pred, y):
    return round(math.sqrt(((pred - y) ** 2).mean()), 6)


def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)


m_rmse(m, xs, y)  # 0.0

# we really need to check the validation set, to ensure we’re not overfitting:
m_rmse(m, valid_xs, valid_y)  # 0.337727

# Oops—it looks like we might be overfitting pretty badly. Here’s why:
m.get_n_leaves(), len(xs)  # (340909, 404710)


# We have nearly as many leaf nodes as data points! That seems a little over-enthusiastic. Indeed,
# sklearn’s default settings allow it to continue splitting nodes until there is only one item in
# each leaf node. Let’s change the stopping rule to tell sklearn to ensure every leaf node
# contains at least 25 auction records:
m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)  # (0.248562, 0.32368) : much better

m.get_n_leaves()  # 12397 : Much more reasonable!

# Random Forests ( best-of-both worlds solution)


# In the following function definition, n_estimators defines the number of trees we
# want, max_samples defines how many rows to sample for training each tree, and
# max_features defines how many columns to sample at each split point (where 0.5
# means “take half the total number of columns”). We can also specify when to stop
# splitting the tree nodes, effectively limiting the depth of the tree, by including the
# same min_samples_leaf parameter we used in the preceding section. Finally, we pass
# n_jobs=-1 to tell sklearn to use all our CPUs to build the trees in parallel.
def rf(
    xs,
    y,
    n_estimators=40,
    max_samples=200_000,
    max_features=0.5,
    min_samples_leaf=5,
    **kwargs,
):
    return RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
    ).fit(xs, y)


m = rf(xs, y)

m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)  # (0.170896, 0.233502)
# Our validation RMSE is now much improved over our last result produced by the
# DecisionTreeRegressor, which made just one tree using all the available data:


# To see the impact of n_estimators, let’s get the predictions from each individual tree in our forest
preds = np.stack([t.predict(valid_xs) for t in m.n_estimators])


r_mse(preds.mean(0), valid_y)  # 0.233502
# As you can see, preds.mean(0) gives the same results as our random forest:

# Let’s see what happens to the RMSE as we add more and more trees.
plt.plot([r_mse(preds[: i + 1].mean(0), valid_y) for i in range(40)])
# As you can see, the improvement levels off quite a bit after around 30 trees


# OOB, Note that we compare them to the training labels, since this is
# being calculated on trees using the training set:
r_mse(m.oob_prediction_, y)  # 0.210686

# get predictions over the validation set using a py list comprehension to do this for each tree in the forest:
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
preds.shape  # (40, 7988) 40 trees and 7,988 auctions

# we can get the standard deviation of the predictions over all the trees, for each auction:
preds_std = preds.std(0)

# Here are the standard deviations for the predictions for the first five auctions
preds_std[:5]
# array([0.21529149, 0.10351274, 0.08901878, 0.28374773, 0.11977206]) (predictions varies widely)


def rf_feat_importance(m, df):
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)


# the first few most important columns have much higher importance scores than the rest
fi = rf_feat_importance(m, xs)
fi[:10]


# A plot of the feature importances shows the relative importances more clearly:
def plot_fi(fi):
    return fi.plot("cols", "imp", "barh", figsize=(12, 7), legend=False)


plot_fi(fi[:30])

# Removing Low-Importance Variables
# It seems likely that we could use a subset of the columns by removing the variables of low
# importance and still get good results. Let’s try keeping just those with a feature
# importance greater than 0.005:
to_keep = fi[fi.imp > 0.005].cols
len(to_keep)  # 21

# We can retrain our model using just this subset of the columns:
xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]
m = rf(xs_imp, y)

m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)
# (0.181208, 0.232323) : Our accuracy is about the same, but we have far fewer columns to study:

len(xs.columns), len(xs_imp.columns)
# (78, 21)

plot_fi(rf_feat_importance(m, xs_imp))

# Removing Redundant Features
# One thing that makes this harder to interpret is that there seem to be some variables
# with very similar meanings: for example, ProductGroup and ProductGroupDesc. Let’s
# try to remove any redundant features.

cluster_columns(xs_imp)


def get_oob(df):
    m = RandomForestRegressor(
        n_estimators=40,
        min_samples_leaf=15,
        max_samples=50000,
        max_features=0.5,
        n_jobs=-1,
        oob_score=True,
    )
    m.fit(df, y)
    return m.oob_score_


get_oob(xs_imp)  # 0.8771039618198545

# Now we try removing each of our potentially redundant variables, one at a time:
{
    c: get_oob(xs_imp.drop(c, axis=1))
    for c in (
        "saleYear",
        "saleElapsed",
        "ProductGroupDesc",
        "ProductGroup",
        "fiModelDesc",
        "fiBaseModel",
        "Hydraulics_Flow",
        "Grouser_Tracks",
        "Coupler_System",
    )
}

# Now let’s try dropping multiple variables. We’ll drop one from each of the tightly
# aligned pairs we noticed earlier. Let’s see what that does:
to_drop = ["saleYear", "ProductGroupDesc", "fiBaseModel", "Grouser_Tracks"]
get_oob(xs_imp.drop(to_drop, axis=1))  # 0.8739605718147015

# Let’s create DataFrames without these columns, and save them:
xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)

(path / "xs_final.pkl").save(xs_final)
(path / "valid_xs_final.pkl").save(valid_xs_final)

xs_final = (path / "xs_final.pkl").load()
valid_xs_final = (path / "valid_xs_final.pkl").load()

# check our RMSE again, to confirm that the accuracy hasn’t substantially changed:
m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)  # (0.183263, 0.233846)

# As we’ve seen, the two most important predictors are ProductSize and YearMade.
# We’d like to understand the relationship between these predictors and sale price

# It’s a good idea to first check the count of values per category
p = valid_xs_final["ProductSize"].value_counts(sort=False).plot.barh()
c = to.classes["ProductSize"]
plt.yticks(range(len(c)), c)

ax = valid_xs_final["YearMade"].hist()

# With these averages, we can then plot each year on the x-axis, and each prediction on
# the y-axis. This, finally, is a partial dependence plot.
fig, ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(
    m, valid_xs_final, ["YearMade", "ProductSize"], grid_resolution=20, ax=ax
)
