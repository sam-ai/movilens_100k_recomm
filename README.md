# movilens_100k_recomm
Movies Recommendation

Recommendation on basis of rating/watching history
- using SVD - ml_svd_model.ipynb
- using Keras embeddings - dl_keras_model.ipynb

Benchmarking different models using surprise python library.
- validating SVD model - ml_svd_surprise_evalute.py

```
pip install -r requirements.txt
python data_download.py

```

Keras Recommendation engine result :

```
Showing recommendations for user: 219
====================================
Movies with high ratings from user
--------------------------------
Wizard of Oz, The (1939) : 01-Jan-1939
Full Monty, The (1997) : 01-Jan-1997
Heathers (1989) : 01-Jan-1989
Paris, Texas (1984) : 01-Jan-1984
Diva (1981) : 01-Jan-1981
--------------------------------
Top 10 movie recommendations
--------------------------------
Usual Suspects, The (1995) : 14-Aug-1995
Shawshank Redemption, The (1994) : 01-Jan-1994
Wrong Trousers, The (1993) : 01-Jan-1993
Empire Strikes Back, The (1980) : 01-Jan-1980
Raiders of the Lost Ark (1981) : 01-Jan-1981
12 Angry Men (1957) : 01-Jan-1957
Amadeus (1984) : 01-Jan-1984
Schindler's List (1993) : 01-Jan-1993
Casablanca (1942) : 01-Jan-1942
Rear Window (1954) : 01-Jan-1954
```
