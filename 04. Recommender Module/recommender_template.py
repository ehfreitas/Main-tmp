import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, ):
        '''
        what do we need to start out our recommender system
        ''' 

    def fit(self, movies_path, reviews_path, latent_features=4, learning_rate=0.0001, iter=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization
        
        INPUT:
        ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate 
        iters - (int) the number of iterations
        
        OUTPUT:
        user_mat - (numpy array) a user by latent feature matrix
        movie_mat - (numpy array) a latent feature by movie matrix
        '''
        movies = pd.read_csv(movies_path)
        reviews = pd.read_csv(reviews_path)

        del movies['Unnamed: 0']
        del reviews['Unnamed: 0']

        self.movies = movies
        self.reviews = reviews

        # Create user-by-item matrix
        user_items = reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

        self.user_movie_df = user_by_movie
        self.ratings_mat = np.matrix(user_by_movie)
        self.users_array = np.array(user_by_movie.index)
        self.movies_array = np.array(user_by_movie.columns)

        # Set up useful values to be used through the rest of the function
        n_users = self.ratings_mat.shape[0]
        n_movies = self.ratings_mat.shape[1]
        num_ratings = np.sum(~np.isnan(self.ratings_mat))
        
        # user matrix filled with random values of shape user x latent 
        user_mat = np.random.rand(n_users, latent_features)
        # movie matrix filled with random values of shape latent x movies
        movie_mat = np.random.rand(latent_features, n_movies)
        
        # initialize sse at 0 for first iteration
        sse_accum = 0
        
        # header for running results
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")
        
        # for each iteration
        for iteration in range(iter):

            # update our sse
            sse_accum = 0
            
            # For each user-movie pair
            for i in range(n_users):
                for j in range(n_movies):
                    # if the rating exists
                    if self.ratings_mat[i, j] > 0:
                        # compute the error as the actual minus the dot product of the user and movie latent features
                        error = self.ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])
                        
                        # Keep track of the total sum of squared errors for the matrix
                        sse_accum += error**2
                        
                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * 2 * error * movie_mat[k, j]
                            movie_mat[k, j] += learning_rate * 2 * error * user_mat[i, k]

            # print results for iteration
            print('{} \t \t {}'.format(iteration, sse_accum/num_ratings))
            
        self.user_mat = user_mat
        self.movie_mat = movie_mat          

    def predict_rating(self, user_id, movie_id):
        '''
        INPUT:
        user_matrix - user by latent factor matrix
        movie_matrix - latent factor by movie matrix
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df
        
        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        # User row and Movie Column
        user_row = np.where(self.users_array == user_id)[0][0]
        movie_col = np.where(self.movies_array == movie_id)[0][0]
        
        # Take dot product of that row and column in U and V to make prediction
        pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])
    
        return pred

    def make_recs(self, _id, _id_type='movie', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        train_data - dataframe of data as user-movie matrix
        train_df - dataframe of training data reviews
        movies - movies df
        rec_num - number of recommendations to return (int)
        user_mat - the U matrix of matrix factorization
        movie_mat - the V matrix of matrix factorization
        
        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id                  
        rec_names - (array) a list or numpy array of recommended movies by name
        '''
        #ranked_df = create_ranked_df(movies, train_df)
        
        if _id_type == 'user' and _id in self.users_array:
            print('Making recommendation for user already in db.')
            user_idx = np.where(self.users_array == _id)[0][0]
            preds = np.dot(self.user_mat[user_idx, :], self.movie_mat)
            bst_movies_idx = preds.argsort()[::-1][:rec_num]
            rec_ids = np.array(self.user_movie_df.columns[bst_movies_idx])
            rec_names = rf.get_movie_names(rec_ids, self.movies)
        elif _id_type == 'movie' and _id in self.movies_array:
            print('Making recommendation for movie already in db.')
            movie_idx = np.where(self.movies_array == _id)[0][0]
            preds = np.dot(self.user_mat, self.movie_mat[:, movie_idx])
            bst_movies_idx = preds.argsort()[::-1][:rec_num]
            rec_ids = np.array(self.user_movie_df.columns[bst_movies_idx])
            rec_names = rf.get_movie_names(rec_ids, self.movies)

        elif _id_type == 'user' and _id not in self.users_array:
            print('User not in db. Making general recommendations.')
            grouped_df = self.reviews.groupby('movie_id')[['user_id', 'rating']].agg({
                'user_id' : 'size', 'rating' : 'mean'
            })
            filtered_df = grouped_df[grouped_df['user_id'] >= 10].sort_values(['rating', 'user_id'], ascending=False).head(rec_num)
            rec_ids = np.array(filtered_df.index)
            rec_names = rf.get_movie_names(rec_ids, self.movies)
        else:
            print('Movie not in db. No recommendations.')
        return rec_ids, rec_names


if __name__ == '__main__':
    # test different parts to make sure it works
    rec = Recommender()

    movies = pd.read_csv('movies_clean.csv')
    reviews = pd.read_csv('train_data.csv')

    del movies['Unnamed: 0']
    del reviews['Unnamed: 0']

    # Create user-by-item matrix
    user_items = reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
    user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

    #print(movies)
    #print(rf.get_movie_names(np.array([454876, 1853728, 1675434, 2125608, 110912]), movies))

    rec.fit('movies_clean.csv', 'train_data.csv', latent_features=15, iter=5, learning_rate=0.005)
    
    print(rec.predict_rating(8, 2844))
    print('\t', rec.make_recs(8, 'user'))
    print('\t', rec.make_recs(1024648, 'movie'))
    print('\t', rec.make_recs(66666666, 'user'))
    