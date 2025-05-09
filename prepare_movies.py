import numpy as np
import pandas as pd
from fancyimpute import IterativeSVD
from sklearn.decomposition import TruncatedSVD

path = ''
print('Loading ratings.dat...')
ratings = pd.read_csv(path + 'ratings.dat', sep='::', engine='python', header=None, usecols=[0,1,2]).values

max_user_id = np.max(ratings[:,0])
max_movie_id = np.max(ratings[:,1])

print('Filling the sparse matrix...')
matrix = np.full((max_user_id, max_movie_id), np.nan)
for row in ratings:
    matrix[row[0] - 1][row[1] - 1] = row[2]

print('Imputing missing values...')
svd_rank = 20
nan_mask = np.isnan(matrix)
matrix_no_nans = matrix
matrix_no_nans[nan_mask] = 0.0
completed_matrix = IterativeSVD(rank = svd_rank).solve(matrix_no_nans, missing_mask = nan_mask)

print('Computing U and V^t...')
tsvd = TruncatedSVD(svd_rank, algorithm="arpack")
U = tsvd.fit_transform(completed_matrix)
VT = tsvd.components_

def save_array_to_text(A, filename):
    f = open(filename, 'w')
    f.write(str(A.shape[0]) + ' ' + str(A.shape[1]) + '\n')
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            f.write("%.7f " % A[i][j])
        f.write('\n')

save_array_to_text(U, path + 'U.txt')
save_array_to_text(VT, path + 'VT.txt')

reconstruction = U.dot(VT)
mae = np.nanmean(abs(matrix - reconstruction))
mse = np.nanmean((matrix - reconstruction) ** 2)
print('Done!')
print('Mean absolute error (for the original values) between X and U*VT: ' + str(mae))
print('Mean square error: ' + str(mse))
