
# Python Imports
import os 

# Package Imports
import pynvml


class DeviceSelector
	def __init__(self, device_selection_string):

		# Init NVML
		pynvml.nvmlInit()	

		#  The device to select 
		self.device = None

		if(device_selection_string == "cuda_auto"):

			# Get the GPU index to use
			device_idx = self._get_gpu_to_use()

			# put it in pytorch format
    		self.device = "cuda:{}".format(device_idx)

		else:
			self.device = device_selection_string

		# Shutdown NVML
		pynvml.nvmlShutdown()

	def get_device(self):
		return self.device

	def _get_gpu_info(self, device):

		# Create the NVML device handle
	    handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
	    
	    # Get the info
	    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
	    num_compute_processes_running = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))

	    return mem_info.free, num_compute_processes_running


	def _get_gpu_to_use(self):

	    # Get the number of devices
	    device_count = torch.cuda.device_count()

	    # Get the labels for each of the devices
	    if("CUDA_VISIBLE_DEVICES" in os.environ):
	        all_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
	        all_devices = [int(s) for s in all_devices]
	    else:
	       all_devices = [i for i in range(device_count)] 

	    # The stats we care about when selecting a GPU
	    free_ram_value = 0
	    num_processes_value = 100000

	    # Select the GPU
	    gpu_select_index = -1

	    # Go through all the devices and
	    for i in range(device_count):

	    	# Get the GPU device ID
	        device_id = all_devices[i]

	        # Get some stats
	        free_ram, num_compute_processes_running = get_gpu_info(device_id)
	        print("GPU {}  has {} MiB free".format(i, (free_ram / (1024*1024))))

	        if(abs(free_ram_value - free_ram) < (500 * 1024 * 1024)):
	        	
	        	# If the GPU doesnt have ram free then select the GPU with the least number of processes
	            if(num_compute_processes_running < num_processes_value):
	                gpu_select_index = i;
	                free_ram_value = free_ram
	                num_processes_value = num_compute_processes_running
	        else:

	        	# Choose the GPU with the most free VRAM
	            if(free_ram_value < free_ram):
	                gpu_select_index = i;
	                free_ram_value = free_ram
	                num_processes_value = num_compute_processes_running

	    return gpu_select_index

