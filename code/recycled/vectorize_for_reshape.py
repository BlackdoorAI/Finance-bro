df = pd.DataFrame(datapoint_list)
df["dur"] = (df["end"] - df["start"]).apply(lambda x:  x.days)
sorted_df = df.sort_values(by='end')
datapoint_list = sorted_df.to_dict(orient= 'records')



df = pd.DataFrame(datapoint_list)
df = df[(df['end'] > useful_ends[0]) & (df['end'] < useful_ends[1])].reset_index()
# pieces = df.to_dict(orient = "records")
#Calculate the boolean matrices of all the combinations and check them
dur_diff = np.abs(df['dur'].values[:, None] - df['dur'].values) < 100 #The duration of the resulting
end_diff = (np.abs((df['end'].values[:, None] - df['end'].values).astype('timedelta64[D]').astype(int)) < 6)
start_diff = (np.abs((df['start'].values[:, None] - df['start'].values).astype('timedelta64[D]').astype(int)) < 6)
# Filter combinations based on your complex conditional logic
# This is a simplified example, you would add more conditions based on your logic above
combined_mask = (dur_diff & (end_diff | start_diff))
np.fill_diagonal(combined_mask, False)
# Identify indices where conditions are met
i, j = np.where(combined_mask)
# Determine whether it is end_diff or start_diff for each pair
condition_type = np.where(end_diff[i, j], 'end', 'start')
# Create a single DataFrame but keep track of which condition was met using 'condition_type'
result = pd.DataFrame({
    'end': np.where(condition_type == 'end', 
                    np.where(df.loc[i, 'dur'].values > df.loc[j, 'dur'].values, df.loc[j, 'start'], df.loc[i, 'start']), 
                    np.where(df.loc[i, 'dur'].values > df.loc[j, 'dur'].values, df.loc[i, 'end'], df.loc[j, 'end'])),
    'start': np.where(condition_type == 'end', 
                    np.where(df.loc[i, 'dur'].values > df.loc[j, 'dur'].values, df.loc[i, 'start'], df.loc[j, 'start']), 
                    np.where(df.loc[i, 'dur'].values > df.loc[j, 'dur'].values, df.loc[j, 'end'], df.loc[i, 'end'])),
    'val': np.where(df.loc[i, 'dur'].values > df.loc[j, 'dur'].values, df.loc[i, 'val'].values - df.loc[j, 'val'].values, df.loc[j, 'val'].values - df.loc[i, 'val'].values),
    'filed': np.maximum(df.loc[i, 'filed'].values, df.loc[j, 'filed'].values),
    'condition': condition_type
})
# Add additional processing based on your logic
# print("before:",len(result.index))
result_wanted = result[np.abs((result['end'] - wanted_end).dt.days) < 6]
print("after:", len(result_wanted.index))
candidates = []
if result_wanted.empty:
    pass
else:
    result_sorted = result_wanted.sort_values(by='filed', ascending=True)
    selected = result_sorted.iloc[0].to_dict()
    selected['special'] = 'synth_combo'
    # Append to a list if needed
    candidates.append(selected)