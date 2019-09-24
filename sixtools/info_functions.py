

###scan info utilities###
def scan_info(scan_id,source='all'):
    ''' 
    Prints to the command line, in a human readable way, the header and/or baseline info for the scan defined by scan_id.
            
    Parameters
    ----------
    scan_id : integer
        The scan_id, or the location form latest scan_id using -1,-2......, to print data from
    source : string, optional
        The source to display info from: can be 'all', 'header' (for header info) or 'baseline' (for baseline info). is 'All' by default.
        
    '''

    if source in ['all','baseline','header']:
        hdr=db[scan_id]
        f_string='*************************************************************************\n'
        f_string+='SCAN NUMBER '+str(hdr.start['scan_id'])+' INFORMATION \n'
        f_string+='*************************************************************************\n\n'        


        if source == 'all' or source == 'header':
            f_string+='HEADER INFORMATION\n------------------\n'
            f_string+='    START INFORMATION\n-----------------\n'
            for key in list(hdr.start.keys()): 
                if key == 'time':
                    f_string+='        time'.ljust(20)+' :'.ljust(4)+str(datetime.datetime.fromtimestamp(hdr.start['time']).strftime('%c'))+'\n'
                elif key != 'scan_id' and key != 'plan_args':
                    f_string+='        '+key.ljust(20)+' :'.ljust(4)+str(hdr.start[key])+'\n'

            f_string+='    STOP INFORMATION\n-----------------\n'
            for key in list(hdr.stop.keys()): 
                if key == 'time':
                    f_string+='        time'.ljust(20)+' :'.ljust(4)+str(datetime.datetime.fromtimestamp(hdr.stop['time']).strftime('%c'))+'\n'
                elif key != 'scan_id' and key != 'plan_args':
                    f_string+='        '+key.ljust(20)+' :'.ljust(4)+str(hdr.stop[key])+'\n'
        
        if source == 'all' or source == 'baseline':
            BL = hdr.table(stream_name='baseline')
            f_string+='BASELINE INFORMATION\n------------------\n'
            
            exit_val=0
            
            keys = list(BL.keys())
            while len(keys) > 0 and exit_val <=50:
                exit_val+=1
                if keys[0] == 'time':
                    device = keys[0]
                else:
                    device = keys[0].partition('_')[0]
                f_string+='    '+str(device)+'\n-----------\n'
                device_keys = list(key for key in keys if key.startswith(device))
                for key in device_keys:

                    f_string+='        '+key.ljust(30)+' start val : '+str(BL[key][1]).ljust(30)+' stop val : '+str(BL[key][2])+'\n'

                keys = list(key for key in keys if not key.startswith(device))
                    
        print (f_string)
  
    else:
        print ('source must be "all", "header" or "baseline"')



### detector utilities ##

def scan_dets(*scan_no):
    '''Enter scan id numbers to print settings for primary stream detectors'''
   
    
    xtr_str = ''
    for scan in scan_no:
        if 'reason' in db[scan].start:
            xtr_str = '   ' + db[scan].start['reason']
        elif 'purpose' in db[scan].start:
            xtr_str = '   ' + db[scan].start['purpose']
        else:
            xtr_str = '   '
            print('Scan ID {}:  {}'.format(scan, xtr_str) )

        for detector in db[scan]['start']['detectors']:
            print('  {} (primary stream):  '.format(detector) )
            settings = db[scan].config_data(detector)['primary'][0]
                
            try:
                for p, n in settings.items():
                    print('    {:_<30} : {:_>20}'.format(p, n))
                print('{:-<70}'.format('-'))
            except:
                pass

