import pandas as pd

df = pd.read_csv('ODI-2018(1).csv', error_bad_lines=False)

all_studies = []

remove_studies = ['duisenberg quantitative risk management', 'drug discovery and safety',
					'21+ac0-05+ac0-1995', 'exchange', 'qrm', 'data mining techniques',
					'ms', 'master human movement science', 'finance', 'quantitative risk management',
					'phd student at fgb', 'phd student', 'mpa', 'mathematics exchange',
					'phd', '+aci-b science', 'economics', 'duisenberg honors program quantitative risk managament',
					'physics', 'mathematics', 'finance dhp qrm']

artificial_intelligence = ['ai', 'a. i. ', 'artificial intelligence', 'artificial', 'intelligence']
bioinformatics = ['bio', 'bioinformatics']
business_analytics = ['ba', 'business analytics', 'analytics']
computer_science = ['cs', 'computer', 'computer science', 'comoputer science']
computational_science = ['cls', 'csl', 'computational', 'computational science']
econometrics = ['econometrics', 'eor', 'or']


def contains_word(w, s):
	for i in range(0, len(s)):
		if (' ' + w + ' ') in (' ' + s[i] + ' '):
			return True
	return False

for index, row in df.iterrows():
	for i in range(0, len(all_studies) + 1):
		if i == len(all_studies):
			all_studies.append([row['What programme are you in?'].lower(), 1])
			break
		if row['What programme are you in?'].lower() == all_studies[i][0]:
			all_studies[i][1] += 1
			break

for i in range(0, len(df)):
	if df['What programme are you in?'][i].lower() in remove_studies:
		df.drop(i, inplace = True)

new_df = df.to_csv("ODI-FILTERED-STUDIES.CSV")

df = pd.read_csv("ODI-FILTERED-STUDIES.CSV", error_bad_lines=False)

for i in range(0, len(df)):
	if contains_word(df['What programme are you in?'][i].lower(), artificial_intelligence) == True or 'artificial' in df['What programme are you in?'][i].lower() or 'ai' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'AI'
	elif contains_word(df['What programme are you in?'][i].lower(), bioinformatics) or 'bio' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'BIO'
	elif contains_word(df['What programme are you in?'][i].lower(), business_analytics) == True or 'analytics' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'BA'
	elif contains_word(df['What programme are you in?'][i].lower(), computer_science) == True or 'computer' in df['What programme are you in?'][i].lower() or 'big data' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'CS'
	elif contains_word(df['What programme are you in?'][i].lower(), computational_science) == True or 'comoputational' in df['What programme are you in?'][i].lower() or 'computational' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'CLS'
	elif contains_word(df['What programme are you in?'][i].lower(), econometrics) == True or 'eco' in df['What programme are you in?'][i].lower():
		s = df['What programme are you in?']
		s.loc[i] = 'EOR'
	else:
		continue

df.to_csv("NEW-ODI-FILTERED-STUDIES.CSV")


print df['What programme are you in?']

# for data in all_studies:
# 	print data
