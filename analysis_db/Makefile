#MAKEFLAGS += -j5

### example : show example how to use this file
.PHONY : example
example :
					@echo "to download data given a batch list; -n is the text flag"
					@echo "make -f make.make batch=batch02 download=batch02_tracks_test.json A01 -n"
					@echo "to make targets without re-downloading the data; -n is the text flag"
					@echo "make -f make.make batch=batch02 download=False B01-n"
					@echo "print targets"
					@echo "make -f make.make batch=batch02 download=False print-B01_targets -n"
					@echo "force target to rebuld -B"
					@echo "make -B -f make.make batch=batch02 download=False B01 -n"

MKDIR_P = mkdir -p

# define paths to analysis
analysisfolder		= $(shell jq -r '.paths.base' ../config/config.json)analysis_db/#/home/mhell/2021_ICESat2_tracks/analysis_db/
plotsfolder	     	= $(shell jq -r '.paths.plot' ../config/config.json)#/home/mhell/2021_ICESat2_tracks/plots/

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

.PHONY : A01
A01 : $(A01_targets)

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

.PHONY : B01 B02
B01 : make_folder $(B01_targets)
B02 : $(B02_targets)

$(B01_targets) : $(work_folder)/B01_regrid_$(hemis)/%_B01_binned.h5 : $(scratch_folder)/$(batch_key)/processed_ATL03_%.h5
					python $(analysisfolder)/B01_filter_regrid.py $* $(batch_key) $(test_flag) > log/B01/$*.txt 2>&1

$(B02_targets) : $(work_folder)/B02_spectra_$(hemis)/B02_%_FFT.nc : $(work_folder)/B01_regrid_$(hemis)/%_B01_binned.h5 $(analysisfolder)/B02_make_spectra.py
					sleep $${RANDOM:0:2}s
					python $(analysisfolder)/B02_make_spectra_gFT.py $* $(batch_key) $(test_flag) > log/B02/$*.txt 2>&1

B_collect : #B01 B02
					${MKDIR_P} $(plotsfolder)$(hemis)/$(batch_key)/B02/
					rm -fv $(plotsfolder)$(hemis)/$(batch_key)/B02/*
					cp $(plotsfolder)$(hemis)/$(batch_key)/*/B_spectra/B02_specs_*.png $(plotsfolder)$(hemis)/$(batch_key)/B02/

#### delete bad tracks ###

# defeins targets for all bad tracks.
bad_beams_files := $(shell ls $(bad_tracks_folder)/$(batch_key)) #$(shell jq -r .[] $(config_folder)config.json)
bad_beams := $(basename $(bad_beams_files))

source_files := $(foreach i, $(bad_beams) ,$(addprefix $(scratch_folder)/$(batch_key), /processed_ATL03_${i}.h5 ) )
#B01_files_01 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_binned.h5 ) )
#B01_files_02 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_corrected.h5 ) )
#B01_files_03 := $(foreach i, $(bad_beams) ,$(addprefix $(work_folder)/B01_regrid_SH/, ${i}_B01_regridded.h5 ) )

plot_folders := $(foreach i, $(bad_beams) ,$(addprefix $(plotsfolder)/$(hemis)/tracks/, ${i} ) )

rm_bad_beams: #rm_B01_files
					@echo "removing bad tracks in 5 sec"
					sleep 5
					rm -fv $(source_files)
					#rm -fv $(B01_targets) $(B01_files_02) $(B01_files_03)
					rm -fv -r $(plot_folders)
					mv $(bad_tracks_folder)/$(batch_key)/*.json $(bad_tracks_folder)/$(batch_key)_deleted/


# #beam_list := $(foreach i, $(beam_list_rar) , $(subst processed_ATL03_,,$(i))  )

# sync plots from batch key to gdrive folder.
sync_gdrive :
					rclone sync $(plotsfolder)$(hemis)/$(batch_key)/B02/ gdrive:Projects/2021_ICESat2_tracks/plots/$(hemis)/$(batch_key)/B02/
					#rclone sync $(plotsfolder)../movie_temp_files/$(key) gdrive:Projects/2020_moist_two_layer/plots/movie_temp_files/$(key)/

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


### Convert file Name to ATL10 file

.PHONY: ATL03_to_ATL10

ATL03_to_ATL10:
	python3 A10_find_ATL10.py ../track_lists/batch02_tracks.json $(hemis)
