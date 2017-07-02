import abc
import datetime
import time
import httplib2
import os
import yaml
import click
import types
import regex
from ortools.constraint_solver import pywrapcp

from oauth2client.file import Storage
from oauth2client import client
from oauth2client import tools
from apiclient import discovery


REFERENCE_TIME = datetime.datetime(2000, 1, 1, 0, 0, 0)


def int_or_none(obj):
    """Tries to convert object into an integer or returns None if not possible"""
    try:
        if isinstance(obj, types.StringTypes):
            return int(obj)
        elif isinstance(obj, list):
            return int(obj[0])
    except ValueError or TypeError:
        return None


class Constraint:
    """Abstract Base class for all constraints """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_constraint(self, solver, time_opt_var, duration_opt_var):
        pass


class DatespanConstraint(Constraint):
    """Constraint for a task to be inbetween two specified days, if no start date is specified, now will be assumed"""
    today = datetime.date.today()

    def __init__(self, start_day, start_month, start_year, end_day, end_month, end_year):
        if start_year is None:
            start_year = DatespanConstraint.today.year
        if end_year is None:
            end_year = DatespanConstraint.today.year
        if start_day is None:
            start_day = DatespanConstraint.today.day
            start_month = DatespanConstraint.today.month
        if end_day is None or end_month is None:
            raise ValueError("Datespan constraint not fully specified.")
        self.start_date = datetime.date(start_year, start_month, start_day)
        self.end_date = datetime.date(end_year, end_month, end_day)
        if self.start_date > self.end_date:
            raise ValueError("Startdate must be the same or before the Enddate")

    def add_constraint(self, solver, time_opt_var, duration_opt_var):
        start_stamp = int((datetime.datetime.combine(self.start_date, datetime.datetime.min.time())
                           - REFERENCE_TIME).total_seconds()/60)
        end_stamp = int((datetime.datetime.combine(self.end_date, datetime.datetime.max.time())
                           - REFERENCE_TIME).total_seconds()/60)
        solver.Add(time_opt_var + duration_opt_var <= end_stamp)
        solver.Add(time_opt_var >= start_stamp)


class TimespanConstraint(Constraint):
    """Constraint for a task to be inbetween two specified times"""
    today = datetime.date.today()

    def __init__(self, start_hour, start_minute, start_ampm, end_hour, end_minute, end_ampm):
        if start_hour is None or end_hour is None:
            raise ValueError("Datespan constraint not fully specified.")
        if start_minute is None:
            start_minute = 0
        if end_minute is None:
            end_minute = 0
        if start_ampm == "am" and start_hour == 12:
            start_hour = 0
        if start_ampm == "pm":
            start_hour = start_hour+12 if start_hour != 12 else start_hour
        if end_ampm == "am" and end_hour == 12:
            end_hour = 0
        if end_ampm == "pm":
            end_hour = end_hour+12 if end_hour != 12 else end_hour
        self.start_time = datetime.time(start_hour, start_minute)
        self.end_time = datetime.time(end_hour, end_minute)
        if self.start_time >= self.end_time:
            raise ValueError("Starttime must be before the Enddate")

    def add_constraint(self, solver, time_opt_var, duration_opt_var):
        start_minutes = self.start_time.hour*60+self.start_time.minute
        end_minutes = self.end_time.hour*60+self.end_time.minute
        solver.Add((time_opt_var + duration_opt_var) % 1440 <= end_minutes)
        solver.Add(time_opt_var % 1440 >= start_minutes)
        solver.Add(time_opt_var % 1440 < (time_opt_var + duration_opt_var) % 1440)


class DurationConstraint(Constraint):
    """Constraint for a task to have an exact length"""
    today = datetime.date.today()

    def __init__(self, hours, minutes):
        if hours is None:
            hours = 0
        elif minutes is None:
            minutes = 0

        self.minutes = hours*60+minutes
        if self.minutes == 0:
            raise ValueError("Duration constraint not fully specified.")

    def add_constraint(self, solver, time_opt_var, duration_opt_var):
        solver.Add(duration_opt_var == self.minutes)


class Task:
    """Representation of a task """
    def __init__(self, description, constraints):
        self.description = description
        self.constraints = constraints
        self.current_duration = None
        self.current_timestamp = None
        self.identifier = None
        self.completed = False

    def is_overdue(self):
        if self.current_timestamp is None:
            return False
        if REFERENCE_TIME + datetime.timedelta(minutes=self.current_timestamp) < datetime.datetime.now():
            return True
        return False

    def add_id(self, identifier):
        self.identifier = identifier

    def __unicode__(self):
        return u'{description}: {current_timestamp} {current_duration} {completed} {identifier}'\
            .format(description = self.description,
                    current_timestamp = self.current_timestamp,
                    current_duration = self.current_duration,
                    completed=self.completed,
                    identifier=self.identifier)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return self.__str__()


class Event:
    """Minimal Representation of a event stored in google calendar"""
    def __init__(self, description, start, end, identifier):
        """start and end can be either date objects for full day events or datetime for timed events"""
        self.description = description
        self.start = start
        self.end = end
        self.identifier = identifier

    def __unicode__(self):
        if isinstance(self.start, datetime.datetime):
            return u'{description}: {start} - {end} ({identifier})'.format(description=self.description,
                                                                           start=self.start.strftime("%d.%m.%Y %H:%m"),
                                                                           end=self.end.strftime("%d.%m.%Y %H:%m"),
                                                                           identifier=self.identifier)
        else:
            return u'{description}: {start} - {end} ({identifier})'.format(description=self.description,
                                                                           start=self.start.strftime("%d.%m.%Y"),
                                                                           end=self.end.strftime("%d.%m.%Y"),
                                                                           identifier=self.identifier)

    def start_timestamp(self):
        if isinstance(self.start, datetime.datetime):
            return int((self.start-REFERENCE_TIME).total_seconds()/60)
        return int((datetime.datetime.combine(self.start, datetime.datetime.min.time())
                    - REFERENCE_TIME).total_seconds()/60)

    def end_timestamp(self):
        if isinstance(self.end, datetime.datetime):
            return int((self.end-REFERENCE_TIME).total_seconds()/60)
        return int((datetime.datetime.combine(self.end, datetime.datetime.min.time())
                    - REFERENCE_TIME).total_seconds()/60)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return self.__str__()


class TaskSpecifierParamType(click.ParamType):
    name = 'task specifier'
    # in all strings which are formatted later regex' {...} must be replaced with {{...}}
    duration = r"( *(for *)?((?P<{hours}>\d+) *h(our)?s?)(( *and *)?(?P<{minutes}>\d+) *m(in)?(ute)?s?)?| " \
               r"*(for *)?((?P<{minutes}>\d+) *m(in)?(ute)?s?)(( *and *)?(?P<{hours}>\d+) *h(our)?s?)?)"
    duration_no_params = r"( *(for *)?((?:\d+) *h(our)?s?)(( *and *)?(?:\d+) *m(in)?(ute)?s?)?| *(for *)?((?:\d+) " \
                         r"*m(in)?(ute)?s?)(( *and *)?(?:\d+) *h(our)?s?)?)"
    date = r"((?P<{day}>[0-3]?[0-9])[.-/](?P<{month}>[01]?[0-9])[./-]?(?P<{year}>\d{{4}})?|(?P<{year}>\d{{4}})[.-/]"  \
           r"(?P<{month}>[01]?[0-9])[.-/](?P<{day}>[0-3]?[0-9])[./-]?)"
    date_no_params = r"((?:[0-3]?[0-9])[.-/](?:[01]?[0-9])[./-]?(?:\d{4})?|(?:\d{4})[.-/](?:[01]?[0-9])[.-/]"  \
                     r"(?:[0-3]?[0-9])[./-]?)"
    time = r"((?P<{hour}>[012]?[0-9])(:(?P<{minute}>[0-6]?[0-9]))?(?P<{ampm}>am|pm)?)"
    time_no_params = r"((?:[012]?[0-9])(:(?:[0-6]?[0-9]))?(?:am|pm)?)"
    datespan = r"(( *(between *)?{start_date} *(and|-) *{end_date})|( *until *{end_date}))"\
        .format(start_date=date.format(day=r"start_day", month=r"start_month", year=r"start_year"),
                end_date=date.format(day=r"end_day", month=r"end_month", year=r"end_year"))
    datespan_no_params = r"(( *(between *)?{date_no_params} *(and|-) *{date_no_params})|( *until *{date_no_params}))"
    timespan = r"(( *(between *)?{start_time} *(and|-) *{end_time}))"\
        .format(start_time=time.format(hour=r"start_hour", minute=r"start_minute", ampm=r"start_ampm"),
                end_time=time.format(hour=r"end_hour", minute=r"end_minute", ampm=r"end_ampm"))
    timespan_no_params = r"(( *(between *)?{time_no_params} *(and|-) *{time_no_params}))"
    long_duration = r"((?P<{days}>\d+)? *(?P<{kind}>d)(ay(s)?)?|(?P<{weeks}>\d+)? *(?P<{kind}>w)(eek(s)?)?|" \
                    r"(?P<{months}>\d+)? *(?P<{kind}>m)(onth(s)?)?|(?P<{years}>\d+)? *(?P<{kind}>y)(ear(s)?))?"\
        .format(days="rep_days", weeks="rep_weeks", months="rep_months", years="rep_years", kind="rep_kind")
    long_duration_no_params = r"((?:\d+)? *(?:d)(ay(s)?)?|(?:\d+)? *(?:w)(eek(s)?)?|" \
                              r"(?:\d+)? *(?:m)(onth(s)?)?|(?:\d+)? *(?:y)(ear(s)?))?"\
        .format(days="rep_days", weeks="rep_weeks", months="rep_months", years="rep_years", kind="rep_kind")
    repetition = r"( *every *{long_duration}( *(?P<{scheduling}>after *completion|in *general))?)"\
        .format(long_duration=long_duration, scheduling="rep_scheduling")
    repetition_no_params = r"( *every *{long_duration_no_params}( *(?:after *completion|in *general))?)"\
        .format(long_duration_no_params=long_duration_no_params)
    overdue_behaviour = r"( *(as)? *(?P<{overdue}>optional|mandatory))"\
        .format(overdue="overdue")
    overdue_behaviour_no_params = r"( *(as)? *(?:optional|mandatory))"
    full_regex = r"({duration}|{datespan}|{timespan}|{repetition}|{overdue_behaviour}| *(?P<description>.+?"\
                 r"(?={duration_no_params}|{datespan_no_params}|{timespan_no_params}|{repetition_no_params}|"\
                 r"{overdue_behaviour_no_params}|$)))+"\
        .format(duration=duration.format(hours="hours", minutes="minutes"), datespan=datespan, timespan=timespan,
                repetition=repetition, overdue_behaviour=overdue_behaviour,
                duration_no_params=duration_no_params,
                datespan_no_params=datespan_no_params.format(date_no_params=date_no_params),
                timespan_no_params=timespan_no_params.format(time_no_params=time_no_params),
                repetition_no_params=repetition_no_params, overdue_behaviour_no_params=overdue_behaviour_no_params)
    re = regex.compile(full_regex)

    def convert(self, value, param, ctx):
        try:
            match = TaskSpecifierParamType.re.fullmatch(value)
            (description, constraints) = TaskSpecifierParamType.extract_match(match)
            return Task(description, constraints)
        except ValueError as ve:
            self.fail('%s is not a valid task specifier\n%s' % (value, ve.message))

    @staticmethod
    def extract_match(match):
        # everything in the string has to be matched
        if match is None:
            raise ValueError("regex didn't result in a fullmatch.")

        # there has to be a description
        if len(match.captures("description")) > 1:
            raise ValueError("description is not specified in one part")
        if len(match.captures("description")) == 0:
            raise ValueError("description is missing")

        # no constraint may be defined twice
        if len(match.captures("start_month")) > 1 or len(match.captures("start_year")) > 1 or\
           len(match.captures("start_day")) > 1 or len(match.captures("end_month")) > 1 or\
           len(match.captures("end_year")) > 1 or len(match.captures("end_day")) > 1:
            raise ValueError("datespan is not clear")
        if len(match.captures("start_hour")) > 1 or len(match.captures("start_minute")) > 1 or\
           len(match.captures("start_ampm")) > 1 or len(match.captures("end_hour")) > 1 or\
           len(match.captures("end_minute")) > 1 or len(match.captures("end_ampm")) > 1:
            raise ValueError("timespan is not clear")
        if len(match.captures("hours")) > 1 or len(match.captures("minutes")) > 1:
            raise ValueError("duration is not clear")

        # there has to be a duration
        if len(match.captures("hours")) != 1 and len(match.captures("minutes")) != 1:
            raise ValueError("duration is missing")

        description = match.group("description")
        constraints = []

        if match.group("end_day") is not None:
            datespan_constraint = DatespanConstraint(int_or_none(match.group("start_day")),
                                                     int_or_none(match.group("start_month")),
                                                     int_or_none(match.group("start_year")),
                                                     int_or_none(match.group("end_day")),
                                                     int_or_none(match.group("end_month")),
                                                     int_or_none(match.group("end_year")))
            constraints.append(datespan_constraint)

        duration_constraint = DurationConstraint(int_or_none(match.group("hours")),
                                                 int_or_none(match.group("minutes")))
        constraints.append(duration_constraint)

        if match.group("end_hour") is not None:
            timespan_constraint = TimespanConstraint(int_or_none(match.group("start_hour")),
                                                     int_or_none(match.group("start_minute")),
                                                     match.group("start_ampm"),
                                                     int_or_none(match.group("end_hour")),
                                                     int_or_none(match.group("end_minute")),
                                                     match.group("end_ampm"))
            constraints.append(timespan_constraint)

        if match.group("rep_kind") is not None:
            print("test")

        return description, constraints


TASKSPECIFIER = TaskSpecifierParamType()


class Planner:
    """Main class organizing the planning of the tasks """
    def __init__(self):
        self.tasks = []
        self.events = []

    def add_task(self, task):
        self.tasks.append(task)
        # TODO: Fit task into google calendar

    def sync_with_google_calendar(self):
        syncer = GoogleCalendarSync()
        # TODO: Read all events, compare with already known events, add new ones, reschedule tasks around it
        new_event = False
        cal_events = syncer.get_cal_events()

        for task in self.tasks:
            if task.identifier is not None and task.completed is False:
                task.completed = True

        for cal_event in cal_events:
            # first check if event is a task
            task = next((task for task in self.tasks if task.identifier == cal_event.identifier), None)
            if task is not None:
                # replan overdue tasks
                task.completed = False
                if task.is_overdue():
                    syncer.remove_task(task)
                    task.identifier = None
                    # TODO: handling for overdue tasks which can't be rescheduled to due constraints
                continue

            event = next((False for event in self.events if event.identifier == cal_event.identifier), None)
            # add new events
            if event is None:
                # TODO: deleted events
                self.events.append(event)
                continue

        # check for completed tasks -> all tasks which have an id and are completed
        for task in self.tasks:
            if task.completed and task.identifier is not None:
                task.identifier = None

        # replan the tasks
        self.plan_tasks()

        # add the tasks
        syncer.add_tasks(self.tasks)

    def plan_tasks(self):
        """Use constraint optimization to find possible configuration fulfilling every constraint"""
        solver = pywrapcp.Solver("task_optimization")
        now = datetime.datetime.now()
        minimum = int((now-REFERENCE_TIME).total_seconds()/60)
        # minutes from now
        task_number = 0
        opt_vars = []
        optimizations = []
        scheduled_tasks = [task for task in self.tasks if task.identifier is not None]
        to_be_scheduled_tasks = [task for task in self.tasks if task.identifier is None and not task.completed]

        # set up the optimization problem
        for task in to_be_scheduled_tasks:
            opt_start = solver.IntVar(minimum, minimum+144000, "opt_start_{number}".format(number=task_number))
            opt_duration = solver.IntVar(0, 1440, "opt_duration_{number}".format(number=task_number))
            for constraint in task.constraints:
                constraint.add_constraint(solver=solver,
                                          time_opt_var=opt_start,
                                          duration_opt_var=opt_duration)

            # no overlap with events from google calendar
            for event in self.events:
                solver.Add(solver.Max(opt_start + opt_duration <= event.start_timestamp(),
                                      opt_start >= event.end_timestamp()) == 1)

            # no overlaps with other to be optimized tasks
            for prev_task_id in range(len(optimizations)):
                solver.Add(solver.Max(opt_start + opt_duration <= optimizations[prev_task_id]['start'],
                                      opt_start >= optimizations[prev_task_id]['start'] +
                                      optimizations[prev_task_id]['duration']) == 1)

            # no overlaps with already scheduled tasks
            for scheduled_task in scheduled_tasks:
                solver.Add(solver.Max(opt_start + opt_duration <= scheduled_task.current_timestamp,
                                      opt_start >= scheduled_task.current_timestamp +
                                      scheduled_task.current_duration) == 1)

            opt_vars += [opt_start, opt_duration]
            optimizations.append({'start': opt_start, 'duration': opt_duration, 'task': task})
            task_number += 1

        # set the decision builder
        db = solver.Phase(opt_vars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)

        # solve the problem
        # TODO: enforce some timelimit on the optimization
        solver.NewSearch(db)

        solver.NextSolution()

        for optimization in optimizations:
            optimization['task'].current_timestamp = optimization['start'].Value()
            optimization['task'].current_duration = optimization['duration'].Value()

        solver.EndSearch()


class GoogleCalendarSync:
    """Provides functions to get events from google calendar"""
    @staticmethod
    def get_credentials():
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """

        # If modifying these scopes, delete your previously saved credentials
        # at ~/.pydo/credentials.json
        scopes = 'https://www.googleapis.com/auth/calendar'
        client_secret_file = 'client_secret.json'
        application_name = 'PyDo'

        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.pydo')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir, 'pydo.json')
        store = Storage(credential_path)
        credentials = None
        if os.path.isfile(credential_path):
            credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(client_secret_file, scopes)
            flow.user_agent = application_name
            credentials = tools.run_flow(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials

    def __init__(self):
        credentials = GoogleCalendarSync.get_credentials()
        http = credentials.authorize(httplib2.Http())
        self._service = discovery.build('calendar', 'v3', http=http)

    def get_cal_events(self):
        # always read everything 100days back in time, assumption: no one does anything relevant 100 days in the past
        # with the assumption of planning approximately 1 month into the future those are 130 relevant days and with
        # a maximum of 2500 results more 19 events per day which should normally be enough, especially if tasks are
        # deleted and/or rescheduled

        from_date = (datetime.datetime.utcnow()-datetime.timedelta(100)).isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = self._service.events().list(
            calendarId='primary', timeMin=from_date, maxResults=2500, singleEvents=True,
            orderBy='startTime').execute()

        full_events = events_result.get('items', [])

        if len(full_events) >= 2400:
            print("Warning: {events} found. 2500 is the maximum. Reduce the time to look into the past, to ensure "
                  "that the most relevant events are fetched")

        events = []

        for full_event in full_events:
            description = full_event["summary"]
            start_date_str = full_event['start'].get('date')
            start_datetime_str = full_event['start'].get('dateTime')
            end_date_str = full_event['end'].get('date')
            end_datetime_str = full_event['end'].get('dateTime')
            start = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_datetime_str is None else \
                datetime.datetime.strptime(start_datetime_str[0:19], "%Y-%m-%dT%H:%M:%S")
            end = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_datetime_str is None else \
                datetime.datetime.strptime(end_datetime_str[0:19], "%Y-%m-%dT%H:%M:%S")
            identifier = full_event["id"]

            events.append(Event(description, start, end, identifier))

        return events

    def add_tasks(self, tasks):
        """Adds the passed task to google calendar"""
        for task in tasks:
            # only add scheduled tasks
            if task.current_duration is None \
                    or task.current_timestamp is None \
                    or task.identifier is not None \
                    or task.completed:
                continue

            # get time object for the current task starting time (for now assuming both are in the same timezone always)
            start_time = REFERENCE_TIME + datetime.timedelta(minutes=task.current_timestamp)
            end_time = REFERENCE_TIME + datetime.timedelta(minutes=task.current_timestamp+task.current_duration)
            start_time_local = time.localtime(time.mktime(start_time.timetuple()))

            # get timezone offset
            is_dst = time.daylight and start_time_local.tm_isdst > 0
            utc_offset = - (time.altzone if is_dst else time.timezone)
            utc_offset_hour = int(utc_offset / 3600)
            utc_offset_minutes = int((abs(utc_offset) % 3600) / 60)

            # convert dates to strings
            start_time_string = start_time.strftime("%Y-%m-%dT%H:%M:00") + "%+03d:%02d" % (utc_offset_hour,
                                                                                           utc_offset_minutes)
            end_time_string = end_time.strftime("%Y-%m-%dT%H:%M:00") + "%+03d:%02d" % (utc_offset_hour,
                                                                                       utc_offset_minutes)

            event = {
                'summary': task.description,
                'start': {
                    'dateTime': start_time_string,
                },
                'end': {
                    'dateTime': end_time_string,
                }
            }
            event = self._service.events().insert(calendarId='primary', body=event).execute()
            task.identifier = event['id']

    def remove_task(self, task):
        self._service.events().delete(calendarId='primary', eventId=task.identifier).execute()


def load_pydo_data():
    """ Loads the pydo.yaml from ~/.pydo, and returns Planner object stored there"""
    print("Load pydo data...")
    home_dir = os.path.expanduser('~')
    pydo_user_dir = os.path.join(home_dir, '.pydo')
    if not os.path.exists(pydo_user_dir):
        os.makedirs(pydo_user_dir)
    pydo_user_path = os.path.join(pydo_user_dir, 'pydo.yaml')
    try:
        with open(pydo_user_path, 'r') as pydo_file:
            return yaml.load(pydo_file)
    except IOError:
        print("No pydo user file found, new one created.")
        return Planner()


def save_pydo_data(planner):
    print("Save pydo data...")
    home_dir = os.path.expanduser('~')
    pydo_user_dir = os.path.join(home_dir, '.pydo')
    if not os.path.exists(pydo_user_dir):
        os.makedirs(pydo_user_dir)
    pydo_user_path = os.path.join(pydo_user_dir, 'pydo.yaml')
    with open(pydo_user_path, 'w') as pydo_file:
        yaml.dump(planner, pydo_file)


@click.group(invoke_without_command=True)
@click.pass_context
def pydo(ctx):
    """Todo App with optimized planning and google calendar integration"""
    if ctx.invoked_subcommand is None:
        print("Pydo without arguments will sync your data with google calendar and replan tasks if necessary, "  
              "use --help for more information.")

        planner = load_pydo_data()
        ctx.obj['planner'] = planner
        # TODO: Sync with google calendar and show agenda
        planner.sync_with_google_calendar()
        save_pydo_data(planner)
    else:
        ctx.obj['planner'] = load_pydo_data()


@pydo.command()
@click.pass_context
@click.argument('task', type=TASKSPECIFIER)
def add(ctx, task):
    """Add a task to pydo."""
    planner = ctx.obj['planner']
    planner.add_task(task)
    save_pydo_data(planner)


# start the actual program if the module is run by itself only
if __name__ == '__main__':
    pydo(obj={})
