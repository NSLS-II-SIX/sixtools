from blueksy import simulators


def list_scans(plan, db):
    '''Print a summary of a plan, with some SIX specific info

    Prints a version of the plan, showing moves that occur outside of runs, and
    potential scan id's for any runs. Note that the scan id's listed assume
    that the first scan has the next scan_id inthe sequence, if there is an
    aborted scan, or if other scans occur first, this may not be true.

    If a 'reason' kwarg is included in the metadata this is also included in
    the printout.

    For example the expected printout for each run should be:
    "scan no XXXX, 'reason', motor1 = Y1, motor2 = Y2, ......."
        where XXXX is the expected scan ID, 'reason' is an optional string
        given by the 'reason' kwarg in he scans metadata and Yi is the value
        that motor1 was set too prior to running the scan.

    A partial reference to this should be made in the startup folder using:

    .. codeblock:: python
       from sixtools import list_scans
       def list_scans = partial(list_scans, db=db)


    Parameters
    ----------
    plan: iterable
        must yield'Msg' objects

    db: databroker.
        The databroker that the scans are being logged in.

    '''

    in_run = 0  # A way to track if we are in a run or not.
    prev_scanID = db[-1].start['scan_id']  # The previous scan ID
    changed_motors = ''  # The value of any changed motors

    for msg in plan:
        if in_run:
            if msg.command == 'close_run':
                in_run = 0
                changed_motors = ''
        else:
            if msg.command == 'open_run':
                in_run = 1
                prev_scanID += 1
                scan_string = 'scan no {}'.format(prev_scanID)
                try:  # If a reason is given in the metadata include it.
                    description = msg.kwargs['reason']
                    description.append(', ')
                except KeyError:
                    description = ''

                print(scan_string + ', ' + description + changed_motors)
            elif msg.command == 'set':
                set_string = msg.obj.name + ' = ' + msg.args
                changed_motors.append(','+set_string)


def check_plan(plan, db, check_limits=True, scan_list=True,
               summarize_plan=False):
    '''This is meant to be a 'complete' collection of all possible 'prechecks'
       for plans.

    This function is meant to be used to apply all possible prechecks for a
    plan. By default it does not use the verbose 'summarize_plan' which shows
    every step, but this can be added by setting the summarize_plan kwarg to
    True.

    A partial reference to this should be made in the startup folder using:

    .. codeblock:: python
       from sixtools import check_plan
       def check_plan = partial(check_plan, db=db)

    Parameters
    ----------
    plan: generator.
        The plan generator that is to be checked.

    db: databroker.
        The databroker that the scans are being logged in.

    check_limits: boolean, optional.
        A boolean that indicates if the limits should be checked for this plan.

    scan_list: boolean, optional.
        A boolean that indicates if the scans should be listed for this plan.

    summarize_plan: boolean, optional.
        A boolean that indicates if the full set of steps should be listed.

    '''

    if check_limits:
        simulators.check_limits(plan)

    if scan_list:
        list_scans(plan, db)

    if summarize_plan:
        simulators.summarize_plan(plan)
