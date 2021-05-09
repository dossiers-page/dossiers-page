---
title: Collaborative Movie Recommendation with Word2vec
layout: single
---

We use recommendations every day from friends. Let it be a movie/ a product you want to try. We take it or discard it based on two things. First is our own experience in the domain (say we liked most of the movies by Tom Hanks. The second is how close we know/think the friend chooses (he/she likes Tom Hanks most of the time).  This is what a recommender system tries to do.  

Be it 
* Google giving search results
* Amazon showing people who bought this item also bought
* Goodreads telling similar books based on your shelves. 
There are multiple ways in which these systems are designed with many parameters, ranging from age, sex, region, history of interactions, what is liked, what is disliked etc. 

One of the most straightforward approaches is measuring the distance between two items (how close they are) based on the users who selected (and liked) those items.  Creating a way to calculate the distance between two entities is what word2vec does. It can make representations of each product in terms of users who used/bought/liked.  There is no complex matrix factorisation or sparse matrix operations involved, very easy to update the system with new users/updates to the products.

To illustrate this, we can take the **MovieLens 25M** database with 62,000 movies rated by 162,000 users.

We can define two movies as similar if the majority of a set of users likes both of them. 

* User1: movie1, movie2, movie3, movie4 
* User2: movie1, movie3, movie5, movie10 
* User3: movie10, movie15,movie24, movie41

Here, movie1, movie3 are possibly similar as it appears with user1 and user2 (and much more users in the list) 
After selecting movies (with bare minimum average rating) and users with approximately 30 to 400 films rated, we can train a word2vec to capture the similarity of movies based on users who liked it.  

After that we can select those users who were not in the training set, take 10 of their most liked movies and predict what would they like.  The number of movies correctly predicted from the rest of their movies would tell us how good the recommender is

![Movie Recommendation Distribution](/assets/images/movie_recommender_distribution_plot.png)


Search your favourite movie and check the recommendations.

<iframe src="https://pythonapps.dossiers.page:9443/" title="Movie Recommender" width='100%' height='1000' frameBorder="0"  allowtransparency="true"></iframe>
