This UVIS_Ozone_Dataset.txt file was generated on 2023-12-20 by Jonathon Mason


GENERAL INFORMATION

1. Title of Dataset: UVIS ozone abundances for 2.5 Mars Years

2. Author Information
	A. Principal Investigator Contact Information
		Name: Jonathon Mason
		Institution: The Open University
		Address: Walton Hall, Milton Keynes
		Email: jon.mason@open.ac.uk

	B. Associate or Co-investigator Contact Information
		Name: Manish Patel
		Institution: The Open University
		Address: Walton Hall, Milton Keynes
		Email: manish.patel@open.ac.uk

	C. Alternate Contact Information
		Name: 
		Institution: 
		Address: 
		Email: 

3. Date of data collection: 2018-03-27 to 2022-12-22

4. Geographic location of data collection: The data was collected by the UVIS spectrometer in orbit around the planet Mars. Coverage is provided in the range latitude 74S to 74N at all longitudes.  

5. Information about funding sources that supported the collection of the data: This work was enabled through UK Space Agency grants ST/V002295/1, ST/V005332/1, ST/X006549/1, ST/Y000234/1 and ST/R003025/1, ST/Y000196/1 and the Science and Technology Facilities Council funding through grant ST/X001180/1.


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: n/a

2. Links to publications that cite or use the data: n/a

3. Links to other publicly accessible locations of the data: n/a

4. Links/relationships to ancillary data sets: The model atmosphere for our retrieval was created using the OpenMars database - Holmes, James; Lewis, Stephen; Patel, Manish (2022). OpenMARS MY34-35 database. The Open University. Dataset. https://doi.org/10.21954/ou.rd.20340999.v2

5. Was data derived from another source? no
	A. If yes, list source(s): 

6. Recommended citation for this dataset: Mason, J. P. & Patel M.R., (2023). The ozone column abundances for 2.5 Mars Years as measured by the NOMAD-UVIS spectrometer. [Dataset]. 


DATA & FILE OVERVIEW

1. File List: 
UVIS_ozone_column_retrieval_dataset_v1.txt: Contains the chronological, geographical and ozone adundance data.

2. Relationship between files, if important: 

3. Additional related data collected that was not included in the current data package: We also obtain the aerosol abundance (dust and water ice), however, we have not included it here to avoid conflict with another publication.



METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
The data was collected by the Ultraviolet and VIsible Spectrometer (UVIS), part of the Nadir and Occultation for MArs Discovery (NOMAD) instrument (Vandaele et al., 2018) aboard the ExoMars Trace Gas orbiter (TGO) has been operating around Mars since March 2018 and has provided near continuous radiometric measurements between 200 nm and 650 nm of the surface and atmosphere (Patel et al. 2017). The inversion of the UVIS radiance spectra around the Hartley band (Hartley, 1881) provides the spatial and temporal (seasonal and diurnal) variation of ozone (O3). Outside the Hartley band measurement of the dust and water ice can be obtained.

2. Methods for processing the data: 
The raw data was processed using the methods described in Mason et al., 2022 and Willame et al., 2022 to produce calibrated radiance spectra from the raw UVIS measurements. We developed a radiative transfer retrieval code using the Discrete Ordinates Radiative Transfer (DISORT) package (Stamnes et al. 2000; Thomas and Stamnes 2002) to perform the inversion of the spectra. The retrieve scheme creates a model atmosphere and simulates the martian radiances in the 220 – 320 nm spectral range. The atmospheric constituents, in this case ozone dust and water ice, are iterated until the model and measured radiances agree. The output from the retrieval is the column abundance of ozone, dust and water ice along with their associated uncertainties.

3. Instrument- or software-specific information needed to interpret the data: The data is stored as a comma deimited text file so any software package similar to, Notepad, Excel, MATLAB or Python will be able to read the data.

4. Standards and calibration information, if appropriate: n/a 

5. Environmental/experimental conditions: n/a

6. Describe any quality-assurance procedures performed on the data: n/a

7. People involved with sample collection, processing, analysis and/or submission: Jonathon Mason, Manish Patel.


DATA-SPECIFIC INFORMATION FOR: UVIS_ozone_column_retrieval_dataset_v1.txt
<repeat this section for each dataset, folder or file, as appropriate>

1. Number of variables: 9

2. Number of cases/rows: 1437585

3. Variable List: 
<list variable name(s), description(s), unit(s)and value labels as appropriate for each>
Variable Name   |               Description                           |     units 
----------------------------------------------------------------------------------------
O3 abund	|   The retrieved ozone column abundance              |  micron-atmospheres
O3 abund uncty	|   The uncertainty in the retrieved ozone value      |  micron-atmospheres
ls              |   A measure of the time of year on Mars             |  degrees
longitude 	|   The east longitude of the measurement             |  degrees
latitude 	|   The latitude of the measurement                   |  degrees
lst             |   The local solar time of the measurement           |  hour
sza		|   Solar Zenith Angle				      |	 degrees
psurf	        |   The surface pressure at the observation location  |	 millibar
flag_lambertian	|   The indicator that a Lambertian surface was used  |  n/a
	   

4. Missing data codes: n/a


5. Specialized formats or other abbreviations used: 
ls - solar longitude
lst - local solar time
sza - solar zenith angle
psurf - surface pressure
O3_abund - ozone column abundance
O3_abund_uncty - ozone column abundance uncertainty



References: 

Mason, J. P. Patel, M. R. Leese, M. R. Hathi, B. G. Willame, Y. Thomas, I. R. ... & Vandaele, A. C. (2022). Removal of straylight from ExoMars NOMAD-UVIS observations. Planetary and Space Science, 218, 105432. 

Patel, M. R. Antoine, P. Mason, J. Leese, M. Hathi, B. Stevens, A. H. ... & Lopez-Moreno, J. J. (2017). NOMAD spectrometer on the ExoMars trace gas orbiter mission: part 2—design, manufacturing, and testing of the ultraviolet and visible channel. Applied optics, 56(10), 2771-2782

Vandaele, A. C., Lopez-Moreno, J. J., Patel, M. R., Bellucci, G., Daerden, F., Ristic, B., ... & NOMAD Team. (2018). NOMAD, an integrated suite of three spectrometers for the ExoMars trace gas mission: Technical description, science objectives and expected performance. Space Science Reviews, 214, 1-47.2

Stamnes, K. Tsay, S. C. Wiscombe, W. & Laszlo, I. (2000). DISORT, a general-purpose Fortran program for discrete-ordinate-method radiative transfer in scattering and emitting layered media: documentation of methodology.

Thomas, G. E. & Stamnes, K. (2002). Radiative transfer in the atmosphere and ocean. Cambridge University Press.

Willame, Y. Depiesse, C. Mason, J. P. Thomas, I. R. Patel, M. R. Hathi, B. ... & Bellucci, G. (2022). Calibration of the NOMAD-UVIS data. Planetary and Space Science, 218, 105504.
Stamnes et al. 2000
Thomas and Stamnes 2002