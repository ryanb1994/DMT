# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

with open('trainingset.csv', 'r') as csvfile:
    df = pd.read_csv(csvfile, dtype={'booking_bool':bool,'srch_destination_id':np.int32, 
                                     'site_id':np.int32, 'visitor_location_country_id':np.int32,
                                     'visitor_hist_starrating':np.float, 'visitor_hist_adr_usd':np.float,
                                     'prop_country_id':np.int32, 'prop_id':np.int32, 
                                     'prop_starrating':np.int32, 'prop_review_score':np.float,
                                     'prop_brand_bool':bool, 'prop_location_score1':np.float,
                                     'prop_location_score2':np.float, 'prop_log_historical_price':np.float,
                                     'price_usd':np.float, 'promotion_flag':bool,\
                                     'srch_length_of_stay':np.int32, 'srch_booking_window':np.int32,
                                     'srch_adults_count':np.int32, 'srch_children_count':np.int32,
                                     'srch_room_count':np.int32, 'srch_saturday_night_bool':bool,
                                     'srch_query_affinity_score':np.float, 'orig_destination_distance':np.float,
                                     'random_bool':bool, 'position':np.int32, 'click_bool':bool,
                                     'booking_bool':bool, 'gross_booking_usd':np.float})
print df.head()

#number of unique stuff
print("Number of unique searches:"),len(df.srch_id.unique())
print("Number of countries:"),len(df.prop_country_id.unique())


#an attempt to vizualize the difference in price between the price history of the client
#and the price of a clicked property - can also be used for booked
df['usd_diff'] = abs(df.price_usd - df.visitor_hist_adr_usd)
df.usd_diff = np.log10(df.usd_diff)
diffUsd = np.array(df.usd_diff.loc[df['click_bool']==True])
diffUsd = diffUsd[~np.isnan(diffUsd)]
plt.hist(diffUsd, range = [0,3])
plt.show()

#an attempt to vizualize the difference in rating between the rating history of the client
#and the rating of a clicked property - can also be used for booked
df['star_diff'] = abs(df.prop_starrating - df.visitor_hist_starrating) 
diffStar = np.array(df.star_diff.loc[df['click_bool']==True])
diffStar = diffStar[~np.isnan(diffStar)]
plt.hist(diffStar, range = [0,5])
plt.show()


end = time.time()
print (end - start)