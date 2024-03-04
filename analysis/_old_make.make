#MAKEFLAGS += -j5

### example : show example how to use this file
.PHONY : example
example :
					@echo "to download data given a batch list; -n is the text flag"
					@echo "make -f make.make batch=batch02 download=batch02_tracks_test.json A01 -n"
					@echo "to make targets without re-downloading the data; -n is the text flag"
					@echo "make -f make.make batch=batch02 download=False B01 -n"
					@echo "print targets"
					@echo "make -f make.make batch=batch02 download=False print-B01_targets -n"
					@echo "force target to rebuld -B"
					@echo "make -B -f make.make batch=batch02 download=False B01 -n"

MKDIR_P = mkdir -p

# define paths to analysis
analysisfolder		= $(shell jq -r '.paths.base' ../config/config.json)analysis_db/#/home/mhell/2021_ICESat2_tracks/analysis_db/
plot_folder	     	= $(shell jq -r '.paths.plot' ../config/config.json)#/home/mhell/2021_ICESat2_tracks/plots/

work_folder				= $(shell jq -r '.paths.work' ../config/config.json)#/work/mhell_work/2021_ICESat2_tracks/
scratch_folder    = $(shell jq -r '.paths.scratch' ../config/config.json)#/scratch/mhell/2021_ICESat2_tracks/
track_lists			  = $(shell jq -r '.paths.base' ../config/config.json)track_lists/#/home/mhell/2021_ICESat2_tracks/track_lists/
track_downloader  = $(shell jq -r '.paths.local_script' ../config/config.json)read-ICESat-2/scripts/#/home/mhell/2021_ICESat2_tracks/modules/read-ICESat-2/scripts/
bad_tracks_folder = $(work_folder)bad_tracks


### parameters
#hemisphere of the analysis, either SH or NH
hemis = SH
# define batch here. defined higher-level structure.
batch = batch02#batch01
#replace_flag			= False
batch_key = $(hemis)_$(batch)

# (False) if not False, this should be a beam/track ID.
# thought for processing single beams ..
beam 	= False

# if not False provide a filename.json read this file and download the data. check that you are on the right batch!!
download=False

# flag for testing. not used yet.
test_flag	= False
### ------

ifeq ($(beam), False)
# load nc file list from batch_key folder
beam_list_json := $(shell ls $(scratch_folder)/$(batch_key)/) #$(shell jq -r .[] $(config_folder)config.json)
# reformatting
beam_list_rar := $(basename $(beam_list_json))
beam_list := $(foreach i, $(beam_list_rar) , $(subst processed_ATL03_,,$(i))  )

#substr_list := $(foreach i, $(beam_list_rar) , $(subst _,'',$(i))  )
#ALT := $(foreach i, $(beam_list_rar) , $(words $(subst _,'',$(i))  ) )
# ALT := $(suffix $(beam_list_json) )
else
beam_list= $(beam)
endif

# beam_list_rar := $(basename $(beam_list_json))
# beam_list := $(foreach i, $(beam_list_rar) , $(subst processed_ATL03_,,$(i))  )

# folders for batch
.PHONY : make_folder
make_folder :
					${MKDIR_P} $(scratch_folder)/$(batch_key)
					${MKDIR_P} $(bad_tracks_folder)/$(batch_key)
					${MKDIR_P} $(bad_tracks_folder)/$(batch_key)_deleted


#### section A
# download data

ifeq ($(download), False) # create target list from beam_list
A01_targets := $(foreach i, $(beam_list) ,$(addprefix $(scratch_folder)/$(batch_key)/, processed_ATL03_${i}.h5 ) )
$(A01_targets) : $(scratch_folder)/$(batch_key)/processed_ATL03_%.h5 :
					touch $(scratch_folder)/$(batch_key)/processed_ATL03_$*.h5

else
# create target list from from file in track list folder. The target downloads the respective keys from the interweb.
A01_track_names := $(shell jq -r .[] $(track_lists)$(download))
A01_tracks_rar := $(basename $(A01_track_names))
A01_tracks := $(foreach i, $(A01_tracks_rar) , $(subst ATL03_,,$(i))  )

A01_targets := $(foreach i, $(A01_tracks) ,$(addprefix $(scratch_folder)/$(batch_key)/, processed_ATL03_${i}.h5 ) )

$(A01_targets) : $(scratch_folder)/$(batch_key)/processed_ATL03_%.h5 :
					python $(track_downloader)/nsidc_icesat2_associated.py --user mhell@ucsd.edu --netrc ~/.netrc --product ATL03 --directory $(scratch_folder)/$(batch_key) -F ATL03_$*.h5
					mv $(scratch_folder)/$(batch_key)/ATL03_$*.h5 $(scratch_folder)/$(batch_key)/processed_ATL03_$*.h5
endif


A02_targets := $(foreach i, $(beam_list) ,$(addprefix $(work_folder)/A02_prior_$(hemis)/, A02b_${i}_hindcast_success.json) )
$(A02_targets) : $(work_folder)/A02_prior_$(hemis)/A02b_%_hindcast_success.json : $(analysisfolder)/A02b_WW3_hindcast_prior.py
					python $(analysisfolder)/A02b_WW3_hindcast_prior.py $* $(batch_key) $(test_flag) > log/A02/$*.txt 2>&1

# download associated ALT10 data
A03_path := $(scratch_folder)/$(batch_key)
# this is based on the sucess of B02 righ now, has to be changed later on...
A03_list = $(foreach i, $(B02_success) , $(subst $(B02_path),$(A03_path)/processed_ATL10_,$(i)) )
A03_targets =$(foreach i, $(A03_list) , $(subst _gFT_x,.h5,$(i)) )

#A03_targets := $(foreach i, $(beam_list) ,$(addprefix $(A03_path)/processed_ATL10_, ${i}.h5 ) )
# khgc
$(A03_targets) : $(A03_path)/processed_ATL10_%.h5 : $(work_folder)B02_spectra_$(hemis)/B02_%_FFT.nc
					python $(track_downloader)/nsidc_icesat2_associated.py --user mhell@ucsd.edu --netrc ~/.netrc --product ATL10 --directory $(scratch_folder)/$(batch_key) -F ATL03_$(subst _003_,_005_,$(subst _004_,_005_,$*)).h5
					mv $(A03_path)/ATL10_$(subst _003_,_005_,$(subst _004_,_005_,$*)).h5 $(A03_path)/processed_ATL10_$(subst _003_,_005_,$(subst _004_,_005_,$*)).h5

#/Users/Shared/Projects/2021_IceSAT2_tracks/data/scratch//SH_batch02/processed_ATL10_20190207001112_06190210_004_01.h5

.PHONY : A01 A02 A03
A01 : $(A01_targets)
A02 : $(A02_targets)
A03 : $(A03_targets)

#### section B ####
# B01 				filters and regridds data, data is saves in binnned .h5 tables.
# B02 				makes LS and FFT spectra
# B_collect		collects the plots in a new folder

# Filter in and make spectra
# not sure if this is needed ..
ifeq ($(download), False)
beam_list := $(beam_list)
else
beam_list := $(A01_tracks)
endif

B01_targets := $(foreach i, $(beam_list) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_binned.h5 ) )
B02_targets := $(foreach i, $(beam_list) ,$(addprefix $(work_folder)/B02_spectra_SH/, B02_${i}_FFT.nc ) )

.PHONY : B01 B02# B03
B01 : make_folder $(B01_targets)
B02 : $(B02_targets)

$(B01_targets) : $(work_folder)/B01_regrid_$(hemis)/%_B01_binned.h5 : $(scratch_folder)/$(batch_key)/processed_ATL03_%.h5 $(analysisfolder)/B01_filter_regrid_segments.py
					python $(analysisfolder)/B01_filter_regrid_segments.py $* $(batch_key) $(test_flag) > log/B01/$*.txt 2>&1

$(B02_targets) : $(work_folder)/B02_spectra_$(hemis)/B02_%_FFT.nc : $(work_folder)/B01_regrid_$(hemis)/%_B01_binned.h5 #$(analysisfolder)/B02_make_spectra.py
					sleep $${RANDOM:0:2}s
					python $(analysisfolder)/B02_make_spectra_gFT.py $* $(batch_key) $(test_flag) > log/B02/$*.txt 2>&1

B_collect : #B01 B02
					${MKDIR_P} $(plot_folder)$(hemis)/$(batch_key)/B02/
					rm -fv $(plot_folder)$(hemis)/$(batch_key)/B02/*
					cp $(plot_folder)$(hemis)/$(batch_key)/*/B_spectra/B02_specs_*.png $(plot_folder)$(hemis)/$(batch_key)/B02/

#### delete bad tracks ###

# defeins targets for all bad tracks.
bad_beams_files := $(shell ls $(bad_tracks_folder)/$(batch_key)) #$(shell jq -r .[] $(config_folder)config.json)
bad_beams := $(basename $(bad_beams_files))

source_files := $(foreach i, $(bad_beams) ,$(addprefix $(scratch_folder)/$(batch_key), /processed_ATL03_${i}.h5 ) )
#B01_files_01 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_binned.h5 ) )
#B01_files_02 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_corrected.h5 ) )
#B01_files_03 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_regridded.h5 ) )

plot_folders := $(foreach i, $(bad_beams) ,$(addprefix $(plot_folder)/$(hemis)/tracks/, ${i} ) )

rm_bad_beams: #rm_B01_files
					@echo "removing bad tracks in 5 sec"
					sleep 5
					rm -fv $(source_files)
					rm -fv -r $(plot_folders)
					mv $(bad_tracks_folder)/$(batch_key)/*.json $(bad_tracks_folder)/$(batch_key)_deleted/

#rm -fv $(B01_targets) $(B01_files_02) $(B01_files_03)


# B03 overview plots
B02_path := $(work_folder)B02_spectra_$(hemis)/B02_
# find sources
B02_success := $(basename $(shell ls $(B02_path)*_gFT_x.nc) )
# B02_fail := $(basename $(shell ls $(B02_path)*_fail.json) )
B03_path := $(plot_folder)$(hemis)/$(batch_key)/
# write targets
B03_list := $(foreach i, $(B02_success) , $(subst $(B02_path),$(B03_path),$(i))  )
B03_targets := $(foreach i, $(B03_list) , $(subst _gFT_x,/B03_success.json,$(i) ) )


B03 : $(B03_targets) B03_mkdir B03_collect

reset_B03 :
					ls $(B03_path)*/B03_success.json
					rm -rf $(B03_path)*/B03_success.json
#.PHONY : B03_collect

B03_mkdir :
					${MKDIR_P} $(B03_collect_path)
					${MKDIR_P} $(B03_collect_path)coord_check

$(B03_targets) : $(B03_path)%/B03_success.json : $(work_folder)B02_spectra_$(hemis)/B02_%_FFT.nc
					python $(analysisfolder)/B03_plot_spectra_ov.py $* $(batch_key) $(test_flag) > log/B03/$*.txt 2>&1

B03_collect_path := $(plot_folder)$(hemis)/$(batch_key)/B03_ov/
# search for all file and move, and rename them similar files
B03_success := $(shell ls $(B03_path)*/B03_success.json)
B03_collect_list := $(foreach i, $(B03_success) , $(subst $(B03_path),$(B03_collect_path),$(i) )  )
#B03_collect_targets := $(foreach i, $(B03_collect_list) , $(subst /B03_success.json,_B03_success.json,$(i)) )
B03_collect_targets := $(foreach i, $(B03_collect_list) , $(subst /B03_success.json,_B03_specs_L25000.png,$(i)) )
B03_collect : $(B03_collect_targets)

$(B03_collect_targets) : $(B03_collect_path)%_B03_specs_L25000.png : $(B03_path)%/B03_success.json
					cp $(B03_path)$*/B03_specs_L25000.0.png $(B03_collect_path)$*_B03_specs_L25000.png
					cp $(B03_path)$*/B03_specs_coord_check.png $(B03_collect_path)coord_check/$*_B03_specs_coord_check.png
#cp $(B03_path)$*/B03_spectra/B03_freq_reconst*.pdf $(B03_collect_path)$*_B03_freq_reconst.pdf



B04_targets := $(foreach i, $(B03_success) , $(subst B03_success,B04_success,$(i)) )

B04 : $(B04_targets)

$(B04_targets) : $(B03_path)%/B04_success.json : $(B03_path)%/B03_success.json $(work_folder)/A02_prior_$(hemis)/A02b_%_hindcast_success.json
					python $(analysisfolder)/B04_angle.py $* $(batch_key) $(test_flag) > log/B04/$*.txt 2>&1

B04_success := $(shell ls $(B03_path)*/B04_success.json)
B04_B05_collect_path := $(plot_folder)$(hemis)/$(batch_key)/B04_B05_angle/
B04_collect_list := $(foreach i, $(B04_success) , $(subst $(B03_path),$(B04_B05_collect_path),$(i) )  )
B04_collect_targets := $(foreach i, $(B04_collect_list) , $(subst /B04_success.json,_B04_marginal_distributions.pdf,$(i)) )

B04_B05_mkdir :
					${MKDIR_P} $(B04_B05_collect_path)

B04_collect : B04_B05_mkdir $(B04_collect_targets)
$(B04_collect_targets) : $(B04_B05_collect_path)%_B04_marginal_distributions.pdf : $(B03_path)%/B04_success.json
				cp $(B03_path)$*/B04_marginal_distributions.pdf $(B04_B05_collect_path)$*_B04_marginal_distributions.pdf

## ---------------- B5

B05_targets := $(foreach i, $(B04_success) , $(subst B04_success,B05_success,$(i)) )

B05 : $(B05_targets)

$(B05_targets) : $(B03_path)%/B05_success.json : $(B03_path)%/B04_success.json
					python $(analysisfolder)/B05_define_angle.py $* $(batch_key) $(test_flag) > log/B04/$*.txt 2>&1

B05_success := $(shell ls $(B03_path)*/B05_success.json)
B05_collect_list := $(foreach i, $(B05_success) , $(subst $(B03_path),$(B04_B05_collect_path),$(i) )  )
B05_collect_targets := $(foreach i, $(B05_collect_list) , $(subst /B05_success.json,_B05_dir_ov.pdf,$(i)) )

B05_collect : B04_B05_mkdir $(B05_collect_targets)
$(B05_collect_targets) : $(B04_B05_collect_path)%_B05_dir_ov.pdf : $(B03_path)%/B05_success.json
				cp $(B03_path)$*/B05_dir_ov.pdf $(B04_B05_collect_path)$*_B05_dir_ov.pdf

## ---------------- B6

B06_targets := $(foreach i, $(B05_success) , $(subst B05_success,B06_success,$(i)) )

B06 : $(B06_targets)

$(B06_targets) : $(B03_path)%/B06_success.json : $(B03_path)%/B05_success.json
					python $(analysisfolder)/B06_correct_separate_var.py $* $(batch_key) $(test_flag) > log/B04/$*.txt 2>&1

B06_success := $(shell ls $(B03_path)*/B06_success.json)
B06_collect_path := $(plot_folder)$(hemis)/$(batch_key)/B06_correction/
B06_collect_list := $(foreach i, $(B06_success) , $(subst $(B03_path),$(B06_collect_path),$(i) )  )
B06_collect_targets := $(foreach i, $(B06_collect_list) , $(subst /B06_success.json,_B06_atten_ov.png,$(i)) )

B06_mkdir :
					${MKDIR_P} $(B06_collect_path)

B06_collect : B06_mkdir $(B06_collect_targets)
$(B06_collect_targets) : $(B06_collect_path)%_B06_atten_ov.png : $(B03_path)%/B06_success.json
				cp $(B03_path)$*/B06_correction/$*_B06_atten_ov.png $(B06_collect_path)$*_B06_atten_ov.png


# sync plots from batch key to gdrive folder.
sync_gdrive :
					rclone sync $(plot_folder)$(hemis)/$(batch_key)/ gdrive_brown:2021_ICESat2_tracks/plots/$(hemis)/$(batch_key)/
					#rclone sync $(plot_folder)../movie_temp_files/$(key) gdrive:Projects/2020_moist_two_layer/plots/movie_temp_files/$(key)/

# for printing variables, used for debugging
print-%  : ; @echo $* = $($*)



# dummy for sending mails when target is done.
## generated logfile at log/station.poc_log.txt
# ts := $(shell /bin/date "+%Y-%m-%d--%H-%M-%S")
#
# SUBJECT	=	"fig-dispoint - job $(ID)"
# EMAIL3	=	mhell@ucsd.edu
# mailer	=	/bin/mail -s $(SUBJECT) $(EMAIL3) < "$(logfile)"
#
# .PHONY : sendmail
# sendmail :
# 					@echo "---- Sendmail ----" >> $(logfile)
# 					$(mailer)
