def time_convertion(time):
	time = int(time)
	second = time % 60
	time = time // 60
	minute = time % 60
	hour = time // 60
	return hour,minute,second

def time_estimation(time_start,cur_time,epoch,max_epoch,step,epoch_len):
	time_passed = cur_time - time_start

	iter_total = epoch_len * max_epoch
	iter_passed = epoch_len * epoch + step
	time_future = (iter_total - iter_passed)/iter_passed*time_passed

	time_passed = time_convertion(time_passed)
	time_future = time_convertion(time_future)
	return time_passed,time_future


