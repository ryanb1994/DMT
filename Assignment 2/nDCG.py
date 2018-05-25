import pandas as pd
import numpy as np

# CHOOSE DATA:

df = pd.read_csv('training-10000.csv')

# Ranking (but is in new in github)

df['rank_rel'] = np.zeros(len(df))
df['rank_rel'] = np.where(df.click_bool == 1, 1, df.rank_rel)
df['rank_rel'] = np.where(df.booking_bool == 1, 5, df.rank_rel)

df = df[['srch_id', 'position', 'rank_rel']]

def dcg(i, rel):
    return (2**rel - 1)/(np.log2(i + 1))

#########################################################################################

# CACULATING DCG

sum_dcg = 0
array_dcg = []

for k in range(0, len(df) - 1):
    if k != len(df) - 1:
        if(df['srch_id'][k] < df['srch_id'][k + 1]):
            sum_dcg += dcg(df['position'][k],df['rank_rel'][k])
            array_dcg.append(sum_dcg)
            sum_dcg = 0
        else:
            sum_dcg += dcg(df['position'][k],df['rank_rel'][k])
    else:
        sum_dcg += dcg(df['position'][k],df['rank_rel'][k])
        array_dcg.append(sum_dcg)
        sum_dcg = 0


##########################################################################################



# CALCULATING IDCG



array_idcg = np.array([])

t_idcg = np.array([])

sum_idcg = 0

# l = 0



for k in range(0, len(df) - 1):

    if k != len(df) - 1:

        if(df['srch_id'][k] < df['srch_id'][k + 1]):

            t_idcg = np.append(t_idcg, df['rank_rel'][k])

            t_idcg = np.sort(t_idcg)[::-1]

            for j in range(0, len(t_idcg) - 1):

                sum_idcg += dcg(j + 1, t_idcg[j])

            array_idcg = np.append(array_idcg, sum_idcg)

            sum_idcg = 0

            t_idcg = []

        else:

            t_idcg = np.append(t_idcg, df['rank_rel'][k])

            

    else:

        t_idcg = np.append(t_idcg, df['rank_rel'][k])

        t_idcg = np.sort(t_idcg)[::-1]

        for j in range(1, len(t_idcg) - 1):

            sum_idcg += dcg(j, t_idcg[j - 1])

        array_idcg = np.append(array_idcg, sum_idcg)





##########################################################################################

print 'nDGC = ', (np.sum(array_dcg)/np.sum(array_idcg))
