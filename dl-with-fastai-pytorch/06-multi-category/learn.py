from fastai.vision.all import *

# ruff: noqa: F405
path = untar_data(URLs.PASCAL_2007)

# inspect the CSV file by reading it into a Pandas DataFrame
df = pd.read_csv(path/'train.csv')
df.head()

# access rows and columns of a DataFrame with the iloc property, as if it were a matrix:
df.iloc[:, 0]
# 0    000005.jpg
# ...
# 5010 009961.jpg
# Name: fname, Length: 5011, dtype: object

df.iloc[0, :]
# Trailing :s are always optional (in numpy, pytorch, pandas, etc.), so this is equivalent:
df.iloc[0]

df['fname']

# You can create new columns and do calculations using columns:
df1 = pd.DataFrame()
df1['a'] = [1, 2, 3, 4]
df1

df1['b'] = [10, 20, 30, 40]
df1['a'] + df1['b']

# 0 11
# 1 22
# 2 33
# 3 44
# dtype: int64


# bind function in python
def say_hello(name, say_what="Hello"): return f"{say_what} {name}."


say_hello('Jeremy'),
say_hello('Jeremy', 'Ahoy!')
# ('Hello Jeremy.', 'Ahoy! Jeremy.')

# We can switch to a French version of that function by using partial::
f = partial(say_hello, say_what="Bonjour")
f("Jeremy"), f("Sylvain")
# ('Bonjour Jeremy.', 'Bonjour Sylvain.')
