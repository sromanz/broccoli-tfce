
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <string>
#include <stdlib.h>

// Position in filename where a NIfTI extension (.nii, .nii.gz, .hdr, .img, .hdr.gz, .img.gz) starts,
// or strlen(filename) if there is none. Only the end of the name is checked, so dots in
// directory names do not matter.
size_t NiftiExtensionPosition(const char* filename)
{
	std::string name(filename);
	const char* extensions[] = {".nii.gz", ".hdr.gz", ".img.gz", ".nii", ".hdr", ".img"};
	for (int e = 0; e < 6; e++)
	{
		std::string ext(extensions[e]);
		if ( (name.size() > ext.size()) && (name.compare(name.size() - ext.size(), ext.size(), ext) == 0) )
		{
			return name.size() - ext.size();
		}
	}
	return name.size();
}

// Output filename for a prefix: keeps an explicit extension, otherwise adds one following
// FSLOUTPUTTYPE as in FSL (NIFTI_GZ -> .nii.gz (default), NIFTI -> .nii, NIFTI_PAIR -> .hdr,
// NIFTI_PAIR_GZ -> .hdr.gz)
std::string NiftiOutputFilename(const char* prefixOrFilename, bool forceDefaultExtension)
{
	std::string name(prefixOrFilename);
	size_t pos = NiftiExtensionPosition(prefixOrFilename);
	if ( (pos < name.size()) && !forceDefaultExtension )
	{
		return name;
	}
	name = name.substr(0, pos);

	std::string extension = ".nii.gz";
	const char* type = getenv("FSLOUTPUTTYPE");
	if (type != NULL)
	{
		std::string t(type);
		if (t == "NIFTI")
		{
			extension = ".nii";
		}
		else if (t == "NIFTI_PAIR")
		{
			extension = ".hdr";
		}
		else if (t == "NIFTI_PAIR_GZ")
		{
			extension = ".hdr.gz";
		}
	}
	return name + extension;
}

void CheckFileExtension(const char* filename, bool& extensionOK, std::string& extension)
{
	std::string name(filename);
	if ( (name.size() > 7) && (name.compare(name.size() - 7, 7, ".nii.gz") == 0) )
	{
		extension = ".nii.gz";
		extensionOK = true;
	}
	else if ( (name.size() > 4) && (name.compare(name.size() - 4, 4, ".nii") == 0) )
	{
		extension = ".nii";
		extensionOK = true;
	}
	else
	{
		size_t slash = name.find_last_of('/');
		size_t dot = name.find('.', (slash == std::string::npos) ? 0 : slash + 1);
		extension = (dot == std::string::npos) ? "" : name.substr(dot);
		extensionOK = false;
	}
}

void CreateFilename(char *& filenameWithExtension, nifti_image* inputNifti, const char* extension, bool CHANGE_OUTPUT_FILENAME, const char* outputFilename)
{
	// Find the dot in the original filename
	if (!CHANGE_OUTPUT_FILENAME)
	{
	    int dotPosition = (int)NiftiExtensionPosition(inputNifti->fname);
    		
	    // Allocate temporary array
	    filenameWithExtension = (char*)malloc(strlen(inputNifti->fname) + strlen(extension) + 1);
		
	    // Copy filename to the dot
	    strncpy(filenameWithExtension,inputNifti->fname,dotPosition);
		filenameWithExtension[dotPosition] = '\0';
	}
	else
	{
	    int dotPosition = (int)NiftiExtensionPosition(outputFilename);
    	
	    // Allocate temporary array
	    filenameWithExtension = (char*)malloc(strlen(outputFilename) + strlen(extension) + 1);
		
	    // Copy filename to the dot
	    strncpy(filenameWithExtension,outputFilename,dotPosition);
		filenameWithExtension[dotPosition] = '\0';
	}
	
	// Add the extension
	strcat(filenameWithExtension,extension);
}

void LowpassFilterRegressor(float* h_LowpassFiltered_Regressor, float* h_Regressor, int DATA_T, int HIGHRES_FACTOR, float TR)
{
	// Allocate memory for lowpass filter
	int FILTER_LENGTH = 151;
	float* h_Filter = (float*)malloc(FILTER_LENGTH * sizeof(float));

	// Create lowpass filter
	int halfSize = (FILTER_LENGTH - 1) / 2;
	double smoothing_FWHM = 150.0;
	double sigma = smoothing_FWHM / 2.354 / (double)TR;
	double sigma2 = 2.0 * sigma * sigma;

	double u;
	float sum = 0.0f;
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		u = (double)(i - halfSize);
		h_Filter[i] = (float)exp(-pow(u,2.0) / sigma2);
		sum += h_Filter[i];
	}

	// Normalize
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		h_Filter[i] /= sum;
	}

	// Convolve regressor with filter
	for (int t = 0; t < (DATA_T * HIGHRES_FACTOR); t++)
	{
		h_LowpassFiltered_Regressor[t] = 0.0f;

		// 1D convolution
		//int offset = -(int)(((float)HRF_LENGTH - 1.0f)/2.0f);
		int offset = -(int)(((float)FILTER_LENGTH - 1.0f)/2.0f);
		for (int tt = FILTER_LENGTH - 1; tt >= 0; tt--)
		{
			if ( ((t + offset) >= 0) && ((t + offset) < (DATA_T * HIGHRES_FACTOR)) )
			{
				h_LowpassFiltered_Regressor[t] += h_Regressor[t + offset] * h_Filter[tt];
			}
			offset++;
		}
	}

	free(h_Filter);
}




void LowpassFilterRegressors(float* h_LowpassFiltered_Regressors, float* h_Regressors, int DATA_T, int HIGHRES_FACTOR, float TR, int NUMBER_OF_REGRESSORS)
{
	// Allocate memory for lowpass filter
	int FILTER_LENGTH = 151;
	float* h_Filter = (float*)malloc(FILTER_LENGTH * sizeof(float));

	// Create lowpass filter
	int halfSize = (FILTER_LENGTH - 1) / 2;
	double smoothing_FWHM = 150.0;
	double sigma = smoothing_FWHM / 2.354 / (double)TR;
	double sigma2 = 2.0 * sigma * sigma;

	double u;
	float sum = 0.0f;
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		u = (double)(i - halfSize);
		h_Filter[i] = (float)exp(-pow(u,2.0) / sigma2);
		sum += h_Filter[i];
	}

	// Normalize
	for (int i = 0; i < FILTER_LENGTH; i++)
	{
		h_Filter[i] /= sum;
	}

	for (int r = 0; r < NUMBER_OF_REGRESSORS; r++)
	{
		// Convolve regressor with filter
		for (int t = 0; t < (DATA_T * HIGHRES_FACTOR); t++)
		{
			h_LowpassFiltered_Regressors[t + r * DATA_T*HIGHRES_FACTOR] = 0.0f;
	
			// 1D convolution
			//int offset = -(int)(((float)HRF_LENGTH - 1.0f)/2.0f);
			int offset = -(int)(((float)FILTER_LENGTH - 1.0f)/2.0f);
			for (int tt = FILTER_LENGTH - 1; tt >= 0; tt--)
			{
				if ( ((t + offset) >= 0) && ((t + offset) < (DATA_T * HIGHRES_FACTOR)) )
				{
					h_LowpassFiltered_Regressors[t + r * DATA_T*HIGHRES_FACTOR] += h_Regressors[t + offset + r * DATA_T*HIGHRES_FACTOR] * h_Filter[tt];
				}
				offset++;
			}
		}
	}

	free(h_Filter);
}

void ConvertFloat2ToFloats(float* Real, float* Imag, cl_float2* Complex, int DATA_W, int DATA_H, int DATA_D)
{
    for (int z = 0; z < DATA_D; z++)
    {
        for (int y = 0; y < DATA_H; y++)
        {   
           for (int x = 0; x < DATA_W; x++)
           {    
               Real[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].x;
               Imag[x + y * DATA_W + z * DATA_W * DATA_H] = Complex[x + y * DATA_W + z * DATA_W * DATA_H].y;
           }
        }
    }
    
}

void FreeAllMemory(void **pointers, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (pointers[i] != NULL)
        {
            free(pointers[i]);
        }
    }
}

void FreeAllNiftiImages(nifti_image **niftiImages, int N)
{
    for (int i = 0; i < N; i++)
    {
		if (niftiImages[i] != NULL)
		{
			nifti_image_free(niftiImages[i]);
		}
    }
}

void ReadBinaryFile(float* pointer, int size, const char* filename, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages)
{
	if (pointer == NULL)
    {
        printf("The provided pointer for file %s is NULL, aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
	}	

	FILE *fp = NULL; 
	fp = fopen(filename,"rb");

    if (fp != NULL)
    {
        fread(pointer,sizeof(float),size,fp);
        fclose(fp);
    }
    else
    {
        printf("Could not open %s , aborting! \n",filename);
        FreeAllMemory(pointers,Npointers);
		FreeAllNiftiImages(niftiImages,Nimages);
        exit(EXIT_FAILURE);
    }
}

void AllocateMemory(float *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t& allocatedMemory, const char* variable)
{
    pointer = (float*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
   		perror ("The following error occurred");
	    printf("Could not allocate host memory for variable %s ! \n",variable);     
	 	FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

void AllocateMemoryInt(unsigned short int *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable)
{
    pointer = (unsigned short int*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
		perror ("The following error occurred");
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

    
void AllocateMemoryFloat2(cl_float2 *& pointer, int size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t allocatedMemory, const char* variable)
{
    pointer = (cl_float2*)malloc(size);
    if (pointer != NULL)
    {
        pointers[Npointers] = (void*)pointer;
        Npointers++;
		allocatedMemory += size;
    }
    else
    {
		perror ("The following error occurred");
        printf("Could not allocate host memory for variable %s ! \n",variable);        
		FreeAllMemory(pointers, Npointers);
		FreeAllNiftiImages(niftiImages, Nimages);
		exit(EXIT_FAILURE);        
    }
}

float mymax(float* data, int N)
{
	float max = -100000.0f;
	for (int i = 0; i < N; i++)
	{
		if (data[i] > max)
			max = data[i];
	}

	return max;
}


float mymin(float* data, int N)
{
	float min = 100000.0f;
	for (int i = 0; i < N; i++)
	{
		if (data[i] < min)
			min = data[i];
	}

	return min;
}

bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename)
{       
	if (data == NULL)
    {
        printf("The provided data pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	
	if (inputNifti == NULL)
    {
        printf("The provided nifti pointer for file %s is NULL, aborting writing nifti file! \n",filename);
		return false;
	}	


    char* filenameWithExtension;
    
    // Add the provided filename extension to the original filename, before the dot
    if (addFilename)
    {
        // Insert before the NIfTI extension at the end of the filename (directories may contain dots)
        int dotPosition = (int)NiftiExtensionPosition(inputNifti->fname);
    
        // Allocate temporary array
        filenameWithExtension = (char*)malloc(strlen(inputNifti->fname) + strlen(filename) + 1);
        if (filenameWithExtension == NULL)
        {
            printf("Could not allocate temporary host memory! \n");      
            return false;
        }
    
        // Copy filename to the dot
        strncpy(filenameWithExtension,inputNifti->fname,dotPosition);
        filenameWithExtension[dotPosition] = '\0';
        // Add the extension
        strcat(filenameWithExtension,filename);
        // Add the rest of the original filename
        strcat(filenameWithExtension,inputNifti->fname+dotPosition);    
    }
        
    // Copy information from input data
    nifti_image *outputNifti = nifti_copy_nim_info(inputNifti);    
    // Set data pointer 
    outputNifti->data = (void*)data;        
    // Set data type to float
    outputNifti->datatype = DT_FLOAT;
    outputNifti->nbyper = 4;    
    
	// Change cal_min and cal_max, to get the scaling right in AFNI and FSL
    int N = inputNifti->nx * inputNifti->ny * inputNifti->nz * inputNifti->nt;
	outputNifti->cal_min = mymin(data,N);
	outputNifti->cal_max = mymax(data,N);

    // Change filename and write
    bool written = false;
    if (addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filenameWithExtension, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }
    }
    else if (!addFilename)
    {
        if ( nifti_set_filenames(outputNifti, filename, checkFilename, 1) == 0)
        {
            nifti_image_write(outputNifti);
            written = true;
        }                
    }    
    
    outputNifti->data = NULL;
    nifti_image_free(outputNifti);

    if (addFilename)
    {
        free(filenameWithExtension);
    } 
        
    if (written)
    {      
        return true;
    }
    else
    {
        return false;
    }                        
}

double GetWallTime()
{
    struct timeval time;
    if (gettimeofday(&time,NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

