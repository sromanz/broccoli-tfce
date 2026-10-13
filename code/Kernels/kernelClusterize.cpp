/*
    BROCCOLI: Software for Fast fMRI Analysis on Many-Core CPUs and GPUs
    Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
    DEALINGS IN THE SOFTWARE.
*/

// Help functions
int Calculate2DIndex(int x, int y, int DATA_W)
{
	return x + y * DATA_W;
}

int Calculate3DIndex(int x, int y, int z, int DATA_W, int DATA_H)
{
	return x + y * DATA_W + z * DATA_W * DATA_H;
}

int Calculate4DIndex(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
	return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}

__kernel void SetStartClusterIndicesKernel(__global unsigned int* Cluster_Indices,
										   __global const float* Data,
										   __global const float* Mask,
										   __private float threshold,
 									       __private int contrast,
										   __private int DATA_W,
										   __private int DATA_H,
										   __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// Threshold data
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold) )
	{
		// Set an unique index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (unsigned int)Calculate3DIndex(x,y,z,DATA_W,DATA_H);
	}
	else
	{
		// Make sure that all other voxels have a higher start index
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = (unsigned int)(DATA_W * DATA_H * DATA_D * 3);
	}
}




bool IsInsideVolume(int x, int y, int z, int DATA_W, int DATA_H, int DATA_D)
{
	if (z < 0)
		return false;
	else if (z >= DATA_D)
		return false;
	else if (y < 0)
		return false;
	else if (y >= DATA_H)
		return false;
	else if (x < 0)
		return false;
	else if (x >= DATA_W)
		return false;
	else
		return true;
}


__kernel void ClusterizeScan(__global unsigned int* Cluster_Indices,
						  	  volatile __global float* Updated,
						  	  __global const float* Data,
						  	  __global const float* Mask,
						  	  __private float threshold,
 							  __private int contrast,
						  	  __private int DATA_W,
						  	  __private int DATA_H,
						  	  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Threshold data
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		unsigned int label1, label2, temp;

		label2 = DATA_W * DATA_H * DATA_D * 3;

		// Original index
		label1 = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

		// z - 1
		if ( IsInsideVolume(x-1,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z-1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z-1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		// z

		if ( IsInsideVolume(x-1,y,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		// z + 1

		if ( IsInsideVolume(x-1,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x-1,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x-1,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y-1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y-1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if ( IsInsideVolume(x+1,y+1,z+1,DATA_W,DATA_H,DATA_D) )
		{
			temp = Cluster_Indices[Calculate3DIndex(x+1,y+1,z+1,DATA_W,DATA_H)];
			if (temp < label2)
			{
				label2 = temp;
			}
		}

		if (label2 < label1)
		{
			Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label2;
			float one = 1.0f;
			atomic_xchg(Updated,one);
		}
	}
}


__kernel void ClusterizeRelabel(__global unsigned int* Cluster_Indices,
						  	  	__global const float* Data,
						  	  	__global const float* Mask,
						  	  	__private float threshold,
								__private int contrast,
						  	  	__private int DATA_W,
						  	  	__private int DATA_H,
						  	  	__private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	// Threshold data
	if ( (Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] == 1.0f) && (Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold) )
	{
		// Relabel voxels
		unsigned int label = Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];
		unsigned int next = Cluster_Indices[label];
		while (next != label)
		{
			label = next;
			next = Cluster_Indices[label];
		}
		Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] = label;
	}
}

__kernel void CalculateClusterSizes(__global unsigned int* Cluster_Indices,
						  	  	    volatile __global unsigned int* Cluster_Sizes,
						  	  	    __global const float* Data,
						  	  	    __global const float* Mask,
						  	  	    __private float threshold,	
									__private int contrast,
						  	  	    __private int DATA_W,
						  	  	    __private int DATA_H,
						  	  	    __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Threshold data
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		unsigned int one = 1;
		// Increment counter for the current cluster index
		atomic_add(&Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]],one);		
	}
}

__kernel void CalculateClusterMasses(__global unsigned int* Cluster_Indices,
						  	  	     volatile __global unsigned int* Cluster_Masses,
						  	  	     __global const float* Data,
						  	  	     __global const float* Mask,
						  	  	     __private float threshold,
									 __private int contrast,
						  	  	     __private int DATA_W,
						  	  	     __private int DATA_H,
						  	  	     __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Threshold data
	if ( Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] > threshold )
	{
		// Increment mass for the current cluster index, done in an ugly way since atomic floats are not supported
		atomic_add( &Cluster_Masses[Cluster_Indices[Calculate3DIndex(x,y,z,DATA_W,DATA_H)]], (unsigned int)(Data[Calculate4DIndex(x,y,z,contrast,DATA_W,DATA_H,DATA_D)] * 10000.0f) );
	}
}

__kernel void CalculateLargestCluster(__global unsigned int* Cluster_Sizes,
								      volatile global unsigned int* Largest_Cluster,
   						  	  	      __private int DATA_W,
									  __private int DATA_H,
									  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	unsigned int cluster_size = Cluster_Sizes[Calculate3DIndex(x,y,z,DATA_W,DATA_H)];

	// Most cluster size counters are zero, so avoid running atomic max for those
	if (cluster_size == 0)
		return;

	atomic_max(Largest_Cluster,cluster_size);
}

__kernel void CalculateTFCEValues(__global float* TFCE_Values,
								  __global const float* Mask,
	  	    					  __private float threshold,
								  __global const unsigned int* Cluster_Indices,
							      __global const unsigned int* Cluster_Sizes,
								  __private int DATA_W,
								  __private int DATA_H,
								  __private int DATA_D)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

	if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
		return;

	if ( Mask[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] != 1.0f )
		return;

	// Check if the current voxel belongs to a cluster
	if ( Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] < (DATA_W * DATA_H * DATA_D) )
	{
		// Get extent of cluster for current voxel
		float clusterSize = (float)Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)]];
		float value = sqrt(clusterSize) * threshold * threshold;

		TFCE_Values[Calculate3DIndex(x,y,z,DATA_W,DATA_H)] += value;
	}
}





__kernel void CalculatePermutationPValuesVoxelLevelInference(__global float* P_Values,
							   	   	   	   	   	  	  	  	 __global const float* Test_Values,
							   	   	   	   	   	  	  	  	 __global const float* Mask,
							   	   	   	   	   	  	  	  	 __global const float* c_Max_Values,
							   	   	   	   	   	  	  	  	 __private int contrast,
							   	   	   	   	   	  	  	  	 __private int DATA_W,
							   	   	   	   	   	  	  	  	 __private int DATA_H,
							   	   	   	   	   	  	  	  	 __private int DATA_D,
							   	   	   	   	   	  	  	  	 __private int NUMBER_OF_PERMUTATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 1.0f )
	{
    	float Test_Value = Test_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)];

    	float sum = 0.0f;
    	for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    	{
    		//sum += Test_Value > c_Max_Values[p];
    		if (Test_Value > c_Max_Values[p])
    		{
    			sum += 1.0f;
    		}
    	}
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = sum / (float)NUMBER_OF_PERMUTATIONS;
	}
    else
    {
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }
}

__kernel void CalculatePermutationPValuesClusterExtentInference(__global float* P_Values,
															   __global const float* Test_Values,
															   __global const unsigned int* Cluster_Indices,
															   __global const unsigned int* Cluster_Sizes,
							   	   	   	   	   	  	  	  	   __global const float* Mask,
							   	   	   	   	   	  	  	  	   __global const float* c_Max_Values,
							   	   	   	   	   	  	  	  	   __private float threshold,
							   	   	   	   	   	  	  	  	   __private int contrast,
							   	   	   	   	   	  	  	  	   __private int DATA_W,
							   	   	   	   	   	  	  	  	   __private int DATA_H,
							   	   	   	   	   	  	  	  	   __private int DATA_D,
							   	   	   	   	   	  	  	  	   __private int NUMBER_OF_PERMUTATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 1.0f )
	{
    	// Check if the current voxel belongs to a cluster
    	if ( Test_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] > threshold )
    	{
    		// Get cluster extent of current cluster
    		float Test_Value = (float)Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)]];

    		float sum = 0.0f;
    		for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    		{
    			if (Test_Value > c_Max_Values[p])
    			{
    				sum += 1.0f;
    			}
    		}
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = sum / (float)NUMBER_OF_PERMUTATIONS;
    	}
    	// Voxel is not part of a cluster, so p-value should be 0
    	else
    	{
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    	}
	}
    else
    {
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }
}

__kernel void CalculatePermutationPValuesClusterMassInference(__global float* P_Values,
														      __global const float* Test_Values,
															  __global const unsigned int* Cluster_Indices,
															  __global const unsigned int* Cluster_Sizes,
							   	   	   	   	   	  	  	  	  __global const float* Mask,
							   	   	   	   	   	  	  	  	  __global const float* c_Max_Values,
							   	   	   	   	   	  	  	  	  __private float threshold,
							   	   	   	   	   	  	  	  	  __private int contrast,
							   	   	   	   	   	  	  	  	  __private int DATA_W,
							   	   	   	   	   	  	  	  	  __private int DATA_H,
							   	   	   	   	   	  	  	  	  __private int DATA_D,
							   	   	   	   	   	  	  	  	  __private int NUMBER_OF_PERMUTATIONS)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_global_id(2);

    if (x >= DATA_W || y >= DATA_H || z >= DATA_D)
        return;

    if ( Mask[Calculate3DIndex(x, y, z, DATA_W, DATA_H)] == 1.0f )
	{
    	// Check if the current voxel belongs to a cluster
    	if ( Test_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] > threshold )
    	{
    		// Get cluster mass of current cluster, divide by 10 000 as 10 000 is multiplied with in the CalculateClusterMasses kernel
    		float Test_Value = ((float)Cluster_Sizes[Cluster_Indices[Calculate3DIndex(x, y, z, DATA_W, DATA_H)]]) / 10000.0f;

    		float sum = 0.0f;
    		for (int p = 0; p < NUMBER_OF_PERMUTATIONS; p++)
    		{
    			if (Test_Value > c_Max_Values[p])
    			{
    				sum += 1.0f;
    			}
    		}
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = sum / (float)NUMBER_OF_PERMUTATIONS;
    	}
    	// Voxel is not part of a cluster, so p-value should be 0
    	else
    	{
    		P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    	}
	}
    else
    {
    	P_Values[Calculate4DIndex(x, y, z, contrast, DATA_W, DATA_H, DATA_D)] = 0.0f;
    }
}







// ---------------------------------------------------------------------------------------
// Threshold-free cluster enhancement (TFCE), FSL randomise compatible
//
// One threshold h is processed with five 1D launches over all voxels:
//   TFCEInitLabels -> TFCEMergeLabels -> TFCECompressLabels -> TFCEClusterSizes -> TFCEAccumulate
// Connected components use union-find with atomic_min (Playne & Hawick 2018), so no iterative
// relabelling and no host synchronisation is needed between thresholds.
// FSL semantics: a voxel is suprathreshold if value > h, only voxels inside the mask and not on
// the outermost layer of the volume take part, cluster size is a voxel count, and each step adds
// size^E * h^H (not multiplied by the step size).
// ---------------------------------------------------------------------------------------

#define TFCE_INVALID_LABEL 0xFFFFFFFFu

unsigned int TFCEFind(volatile __global unsigned int* Labels, unsigned int x)
{
	unsigned int parent = Labels[x];
	while (parent != x)
	{
		x = parent;
		parent = Labels[x];
	}
	return x;
}

void TFCEUnion(volatile __global unsigned int* Labels, unsigned int a, unsigned int b)
{
	bool done = false;
	while (!done)
	{
		a = TFCEFind(Labels, a);
		b = TFCEFind(Labels, b);
		if (a < b)
		{
			unsigned int old = atomic_min(&Labels[b], a);
			done = (old == b);
			b = old;
		}
		else if (b < a)
		{
			unsigned int old = atomic_min(&Labels[a], b);
			done = (old == a);
			a = old;
		}
		else
		{
			done = true;
		}
	}
}

__kernel void TFCEInitLabels(__global unsigned int* Labels,
							 __global unsigned int* Cluster_Sizes,
							 __global const float* Data,
							 __global const float* Mask,
							 __private float threshold,
							 __private int contrast,
							 __private int DATA_W,
							 __private int DATA_H,
							 __private int DATA_D)
{
	int idx = get_global_id(0);
	int N = DATA_W * DATA_H * DATA_D;
	if (idx >= N)
		return;

	Cluster_Sizes[idx] = 0;

	int x = idx % DATA_W;
	int y = (idx / DATA_W) % DATA_H;
	int z = idx / (DATA_W * DATA_H);

	bool interior = (x > 0) && (x < DATA_W - 1) && (y > 0) && (y < DATA_H - 1) && (z > 0) && (z < DATA_D - 1);

	if ( interior && (Mask[idx] == 1.0f) && (Data[idx + contrast * N] > threshold) )
	{
		Labels[idx] = (unsigned int)idx;
	}
	else
	{
		Labels[idx] = TFCE_INVALID_LABEL;
	}
}

// MAX_SQUARED_DISTANCE = 1 for 6-connectivity, 2 for 18-connectivity, 3 for 26-connectivity
__kernel void TFCEMergeLabels(volatile __global unsigned int* Labels,
							  __private int MAX_SQUARED_DISTANCE,
							  __private int DATA_W,
							  __private int DATA_H,
							  __private int DATA_D)
{
	int idx = get_global_id(0);
	if (idx >= DATA_W * DATA_H * DATA_D)
		return;

	if (Labels[idx] == TFCE_INVALID_LABEL)
		return;

	// Only active voxels are interior, so all neighbours are inside the volume.
	// Only the "backward" half of the neighbourhood is visited, each pair is merged once.
	for (int dz = -1; dz <= 0; dz++)
	{
		for (int dy = -1; dy <= 1; dy++)
		{
			for (int dx = -1; dx <= 1; dx++)
			{
				if ( (dz == 0) && ( (dy > 0) || ((dy == 0) && (dx >= 0)) ) )
					continue;

				int d2 = dx * dx + dy * dy + dz * dz;
				if (d2 > MAX_SQUARED_DISTANCE)
					continue;

				int neighbour = idx + dx + dy * DATA_W + dz * DATA_W * DATA_H;
				if (Labels[neighbour] != TFCE_INVALID_LABEL)
				{
					TFCEUnion(Labels, (unsigned int)idx, (unsigned int)neighbour);
				}
			}
		}
	}
}

__kernel void TFCECompressLabels(volatile __global unsigned int* Labels,
								 __private int DATA_W,
								 __private int DATA_H,
								 __private int DATA_D)
{
	int idx = get_global_id(0);
	if (idx >= DATA_W * DATA_H * DATA_D)
		return;

	if (Labels[idx] == TFCE_INVALID_LABEL)
		return;

	Labels[idx] = TFCEFind(Labels, (unsigned int)idx);
}

__kernel void TFCEClusterSizes(__global const unsigned int* Labels,
							   volatile __global unsigned int* Cluster_Sizes,
							   __private int DATA_W,
							   __private int DATA_H,
							   __private int DATA_D)
{
	int idx = get_global_id(0);
	if (idx >= DATA_W * DATA_H * DATA_D)
		return;

	unsigned int label = Labels[idx];
	if (label == TFCE_INVALID_LABEL)
		return;

	atomic_inc(&Cluster_Sizes[label]);
}

// HEIGHT_TERM = h^H (computed on the host in single precision, as FSL does)
__kernel void TFCEAccumulate(__global float* TFCE_Values,
							 __global const unsigned int* Labels,
							 __global const unsigned int* Cluster_Sizes,
							 __private float E,
							 __private float HEIGHT_TERM,
							 __private int DATA_W,
							 __private int DATA_H,
							 __private int DATA_D)
{
	int idx = get_global_id(0);
	if (idx >= DATA_W * DATA_H * DATA_D)
		return;

	unsigned int label = Labels[idx];
	if (label == TFCE_INVALID_LABEL)
		return;

	float size = (float)Cluster_Sizes[label];
	float extentTerm;
	if (E == 0.5f)
	{
		extentTerm = sqrt(size);
	}
	else if (E == 1.0f)
	{
		extentTerm = size;
	}
	else
	{
		extentTerm = pow(size, E);
	}

	TFCE_Values[idx] += extentTerm * HEIGHT_TERM;
}

// Exact maximum of a float volume inside a mask. Floats are mapped to order-preserving unsigned
// keys so that atomic_max can be used; the host decodes the key (see DecodeOrderedFloatKey).
__kernel void TFCEMaxOrderedKey(volatile __global unsigned int* Max_Key,
								__global const float* Volume,
								__global const float* Mask,
								__private int contrast,
								__private int DATA_W,
								__private int DATA_H,
								__private int DATA_D)
{
	int idx = get_global_id(0);
	int N = DATA_W * DATA_H * DATA_D;
	if (idx >= N)
		return;

	if (Mask[idx] != 1.0f)
		return;

	unsigned int u = as_uint(Volume[idx + contrast * N]);
	unsigned int key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
	atomic_max(Max_Key, key);
}
