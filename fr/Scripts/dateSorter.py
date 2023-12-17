# %%
month_df_list = []
day_df_list   = []
hour_df_list  = []

months = ['January','February','March', 'April', 'May','June',
          'July', 'August', 'September', 'October', 'November', 'December']

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for month in months:
    temp_df = qa.loc[(qa['Month'] == month)]
    month_df_list.append(temp_df)

for day in days:
    temp_df = qa.loc[(qa['Weekday'] == day)]
    day_df_list.append(temp_df)

for hour in range(24):
    temp_df = qa.loc[(qa['Hour'] == hour)]
    hour_df_list.append(temp_df)