import numpy as np
import xarray as xr
import pandas as pd
from function_rr import calc_rr_mix
import os, gzip, pickle


nat_flag = 0 #1 for nat later period, 2 for nat full period, 0 for early all period, 4 for nat last 10
all_flag = 1 #0 use decade around event, 1 use GWL, 2 use +3C
anom_flag = 1
gwl_flag = 2 #0 use best, 1 use lower bound, 2 use upper bound
best_pred_flag = 1
new_flag = 0
same_counts = 0 #for DAMIP, match all and nat ensembles


anst = '' if anom_flag==0 else '_anom'
lst = '_log'
lla = ['','_p5','p95']
			
if all_flag==1:
	ast3 = lla[gwl_flag]
	ast2 = 'all-gwl'+ast3
	dfy = pd.read_pickle('factual_period_years'+ast3+'.pkl')
elif all_flag==2:
	ast2 = 'all-3C'
	yrba = [2045,2054]
else:
	yrba = [2019,2028]
	ast2 = 'all-event'

if nat_flag==1:
	yrbn = [1996,2020]
	nst = 'nat'
	nst2 = nst+'-'+str(yrbn[0])+'-'+str(yrbn[1])
	match = 0
elif nat_flag==2:
	yrbn = [1950,2020]
	nst = 'nat'
	nst2 = nst+'-'+str(yrbn[0])+'-'+str(yrbn[1])
	match = 0
elif nat_flag==4:
	yrbn = [2011,2020]
	nst = 'nat'
	nst2 = nst+'-'+str(yrbn[0])+'-'+str(yrbn[1])
	match = 1
else:
	yrbn = [1950,1959]
	nst = 'early'
	nst2 = 'nat-earlyALL'
	match = 1

datas = ['CanLEAD-FWI','CMIP6-2deg_DAMIP','CMIP6-HighResMIP']
ast = {'CanLEAD-FWI':'','CMIP6-2deg_DAMIP':'_all','CMIP6-HighResMIP':''}

regions = ['Boreal Cordillera', 'Boreal Plains',
 'Boreal Shield East', 'Boreal Shield West', 'Hudson Plains',
 'Montane Cordillera', 'Pacific Maritime',
 'Taiga Cordillera', 'Taiga Plains', 'Taiga Shield East', 'Taiga Shield West']


if best_pred_flag==1:
	reg_mod = pd.read_csv(file_best_regression_predictors)
else:
    reg_mod = pd.read_csv(file_full_regression_results)

dfa = pd.read_csv(file_areaburned_byecozone)

for r in regions:
	r1 = r.replace(' ','_').lower()
	ab1 = dfa.loc[dfa['ECOZONE']==r,'area_calib_sum'].values[0]
	th = np.log10(ab1)
	
	if best_pred_flag == 1:
		ind2 = [reg_mod.loc[reg_mod['ECOZONE']==r,'VAR1'].values[0]]
	else:
		dfr = reg_mod.loc[reg_mod['ECOZONE']==r1]
		dfrs = dfr.sort_values('R2',ascending=False)
		ind2 = list(dfrs['VAR1'].values)
		if 'FWI7X' not in ind2:
			ind2.append('FWI7X')
		ind2.append('DSRsum')
		ind2.append('DSRJJA')

	for ind1 in ind2:
	
		for ids in datas:
			
			mst = ''

			if same_counts==1 and ids=='CMIP6-2deg_DAMIP':
				countsst = '_samecounts'
				with gzip.open('DAMIP_real_counts.pkl') as f:
					damip_counts = pickle.load(f)
			else:
				countsst = ''

			file_out = insert_path_and_name_to_file_out
			if new_flag==1:
				if os.path.isfile(file_out)==1:
					continue
			file1 = file_area_burned_from_dataset
			if os.path.isfile(file1)==0:
				print(file1)
				continue
			df1r = xr.open_dataset(file1)
			df1r = df1r.transpose('time','ens')

			if all_flag==2:
				if ids!='CanLEAD-FWI':
					continue
	
			if nat_flag>=1:
				if ids!='CMIP6-2deg_DAMIP':
					continue
				df3r = xr.open_dataset(file_area_burned_nat)
				data0 = df3r.sel(time=slice(yrbn[0],yrbn[1]))
			else:
				data0 = df1r.sel(time=slice(yrbn[0],yrbn[1]))
	
	
			if ids=='CanLEAD-FWI':
				ens = []
				if all_flag==0 or all_flag==2:
					data1 = df1r.sel(time=slice(yrba[0],yrba[1]))
				else:
					data1 = df1r.sel(time=slice(dfy.loc['CanESM2']['yn'],dfy.loc['CanESM2']['yx']))
				data1m = data1['area_burned'].values
				data1s = data1['se_pred'].values
			else:
				ens = df1r['ens'].data
				if all_flag==0:
					data1 = df1r.sel(time=slice(yrba[0],yrba[1]))
					data1m = data1['area_burned'].values
					data1s = data1['se_pred'].values
				else:
					data1m = []
					data1s = []
					ens = []
					for ie in range(len(df1r['ens'])):
						a = df1r.isel(ens=ie)
						data_ens = a.sel(time=slice(dfy.loc[a['ens'].data]['yn'],dfy.loc[a['ens'].data]['yx']))
						if len(data_ens['time'])==0:
							data1m.append([np.nan])
							data1s.append([np.nan])
						else:
							data1m.append(data_ens['area_burned'].values)
							data1s.append(data_ens['se_pred'].values)
						ens.append(a['ens'].item())	
					if same_counts==1 and ids=='CMIP6-2deg_DAMIP':
						data1mb = []; data1sb = []
						for jm in damip_counts['all']:
							qm1 = [data1m[h] for h in range(len(ens)) if ens[h]==jm]
							qs1 = [data1s[h] for h in range(len(ens)) if ens[h]==jm]
							if damip_counts['all'][jm]>0:
								data1mb.append(qm1[:damip_counts['all'][jm]]) #select only designated number of members
								data1sb.append(qs1[:damip_counts['all'][jm]]) #select only designated number of members
						data1m = np.hstack([np.hstack(h) for h in data1mb])
						data1s = np.hstack([np.hstack(h) for h in data1sb])
					else:
						ens_len = [len(h) for h in data1m]
						data1m = np.hstack(data1m)
						data1s = np.hstack(data1s)

			data0 = data0.transpose('time','ens')
			data0m = data0['area_burned'].values
			data0s = data0['se_pred'].values
			if same_counts==1 and ids=='CMIP6-2deg_DAMIP':
				data0ma = data0m; ens0=data0['ens'].data
				data0sa = data0s
				data0sb = []; data0mb = []
				for jm in damip_counts['nat']:
					q1m = data0ma[:,ens0==jm]
					q1s = data0sa[:,ens0==jm]
					if damip_counts['nat'][jm]>0:
						data0mb.append(q1m[:,:damip_counts['nat'][jm]])
						data0sb.append(q1s[:,:damip_counts['nat'][jm]])
				data0m = np.hstack(data0mb).flatten()
				data0s = np.hstack(data0sb).flatten()
			elif ids!='CanLEAD-FWI':
				data0mt = []; data0st = []
				for h in range(len(data0m[0,:])):
					ind0 = np.random.choice(range(len(data0m[:,h])),ens_len[h],replace=False)
					#select only as many years from counterfactual as were for factual, but randomly choose
					#make sure mean and std pairs stay together
					data0mt.append(data0m[ind0,h])
					data0st.append(data0s[ind0,h])
				data0m = np.hstack(data0mt)
				data0s = np.hstack(data0st)

	
			rr_dict = calc_rr_mix(data0m,data0s,data1m, data1s, th, match_pairs=match)

			with gzip.open(file_out, 'w') as f:
				pickle.dump(rr_dict,f)
