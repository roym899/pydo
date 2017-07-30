import abc
import datetime
import time
import httplib2
import os
import yaml
import click
import types
import regex
import calendar
from ortools.constraint_solver import pywrapcp

from oauth2client.file import Storage
from oauth2client import client
from oauth2client import tools
from apiclient import discovery


REFERENCE_TIME = datetime.datetime(2000, 1, 1, 0, 0, 0)
MAX_RELAXATION = 5


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
    def add_constraint(self, solver, time_opt_var, duration_opt_var, relaxation):
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

    def add_constraint(self, solver, time_opt_var, duration_opt_var, relaxation=0):
        # add one day per 2 relaxation levels
        added_days = datetime.timedelta(days=relaxation // 2)
        start_stamp = int((datetime.datetime.combine(self.start_date-added_days, datetime.datetime.min.time())
                           - REFERENCE_TIME).total_seconds()/60)
        end_stamp = int((datetime.datetime.combine(self.end_date+added_days, datetime.datetime.max.time())
                           - REFERENCE_TIME).total_seconds()/60)
        solver.Add((time_opt_var + duration_opt_var <= end_stamp) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)
        solver.Add((time_opt_var >= start_stamp) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)


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

    def add_constraint(self, solver, time_opt_var, duration_opt_var, relaxation=0):
        start_minutes = self.start_time.hour*60+self.start_time.minute
        end_minutes = self.end_time.hour*60+self.end_time.minute
        # relaxation, add one hour at start and end per relaxation level
        start_minutes -= relaxation*60
        if start_minutes < 0:
            start_minutes = 0
        end_minutes += relaxation*60
        if start_minutes < 0:
            end_minutes = 0
        solver.Add(((time_opt_var + duration_opt_var) % 1440 <= end_minutes) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)
        solver.Add((time_opt_var % 1440 >= start_minutes) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)
        solver.Add((time_opt_var % 1440 < (time_opt_var + duration_opt_var) % 1440) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)


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

    def add_constraint(self, solver, time_opt_var, duration_opt_var, relaxation=0):
        solver.Add((duration_opt_var == self.minutes) + (time_opt_var == 0)*(duration_opt_var == 0) >= 1)


class Task:
    """Representation of a task """
    def __init__(self, description, constraints, recurring=None, overdue_behaviour='mandatory'):
        self.description = description
        self.constraints = constraints
        self.subtasks = []
        if recurring is None:
            self.subtasks.append({'identifier': None, 'completed': False, 'current_timestamp': None, 'current_duration': None})
        self.recurring = recurring
        self.overdue_behaviour = overdue_behaviour

    def is_recurring(self):
        if not self.recurring:
            return False
        else:
            return True

    @staticmethod
    def is_subtask_overdue(subtask):
        return REFERENCE_TIME + datetime.timedelta(minutes=subtask['current_timestamp']) < datetime.datetime.now()

    def is_overdue(self):
        """Returns a list of True and False depending on which timestamps have passed already"""
        if len(self.subtasks) == 0:
            return [False]

        return [REFERENCE_TIME + datetime.timedelta(minutes=subtask['current_timestamp']) < datetime.datetime.now()
                for subtask in self.subtasks]

    def add_id(self, identifier, subtask_number):
        self.subtasks[subtask_number]['identifier'] = identifier

    def get_completed_subtasks(self):
        return [subtask for subtask in self.subtasks if subtask['completed']]

    def to_be_scheduled_tasks(self):
        """Generates the number of tasks to be scheduled at any time"""

        # non recurring tasks are always scheduled once at max
        if not self.is_recurring():
            return 1
        # recurring tasks are scheduled depending on the recurring time frame
        else:
            # get approximate days between two events
            if self.recurring['kind'] == 'days':
                approx_days = self.recurring['amount']
            else:
                approx_days = 30*self.recurring['amount']

            # 7 events for daily tasks, 2 are minimum for yearly tasks, monthly ~3
            number_of_tasks = int(round(-5*(1-pow(2, (-approx_days+1)/30))+7))
            return number_of_tasks

    def get_next_schedule_date(self, previous_date):
        """ Calculates the next date this task should be generated, none if no date is suitable """
        if self.is_recurring():
            if self.recurring['kind'] == 'month':
                # after completion with a certain day a month scheduling (same for in general and after completion)
                # -> start at start_date and search for a valid date
                current_date = self.recurring['start_date']
                current_year = current_date.year
                current_month = current_date.month

                while self.recurring['end_date'] is None or current_date <= self.recurring['end_date']:
                    if current_date > previous_date:
                        return current_date

                    current_month += self.recurring['amount']
                    current_year += current_month // 12
                    current_month = 1 if current_month % 12 == 0 else current_month % 12
                    if self.recurring['date'] < 0:
                        current_day = calendar.monthrange(current_year, current_month) + self.recurring['date'] + 1
                    else:
                        current_day = self.recurring['date']

                    current_date = datetime.date(year=current_year, month=current_month, day=current_day)

                return None

            if self.recurring['kind'] == 'days':
                if self.recurring['scheduling'] == 'in_general':
                    current_date = self.recurring['start_date']
                    while self.recurring['end_date'] is None or current_date <= self.recurring['end_date']:
                        if current_date > previous_date:
                            return current_date

                        current_date = current_date + datetime.timedelta(days = self.recurring['amount'])

                    return None

                if self.recurring['scheduling'] == 'after_completion':
                    # just add the days to the previous date
                    next_date = previous_date + datetime.timedelta(days = self.recurring['amount'])
                    if self.recurring['end_date'] is not None and next_date > self.recurring['end_date']:
                        return None
                    else:
                        return next_date

        else:
            return None


    def get_nth_schedule_date(self, n):
        """ Calculates the nth date this task should be generated starting from 1 and strom the start date,
         returns none if no date is suitable """
        if self.is_recurring():
            # for mandatory tasks the next date only depends on the number of completed and scheduled subtasks
            # no further params needed
            counter = 0
            previous_date = REFERENCE_TIME.date()
            while counter <= n:
                previous_date = self.get_next_schedule_date(previous_date)
                counter += 1
        else:
            return None

    def add_to_solver(self, solver, task_number, done_optimizations, relaxation):
        """Adds the task and its inherent constraints to the solver, returns object including the optimization
        variables"""
        optimizations = []
        if self.is_recurring():
            to_be_scheduled_tasks = self.to_be_scheduled_tasks()
            scheduled_tasks = 0
            subtask_id = 0
            now = datetime.datetime.now()
            minimum = int((now-REFERENCE_TIME).total_seconds()/60)

            if self.recurring['scheduling'] == 'after_completion':
                reschedule = any([Task.is_subtask_overdue(subtask) for subtask in self.subtasks])
            else:
                reschedule = False

            while scheduled_tasks < to_be_scheduled_tasks:
                to_be_scheduled = False
                overdue = False

                # check one subtaskid after another
                # possible outcomes: already in optimized / does not exist yet / scheduled in the future / overdue / completed

                # POSSIBILITY 0 check if this task is already in the done_optimizations
                if len([done_optimization for done_optimization in done_optimizations
                        if done_optimization['task'] == self and done_optimization['subtask_number'] == subtask_id]) > 0:
                    scheduled_tasks += 1
                    task_number += 1
                # POSSIBILITY 1 subtask does not exist yet
                elif subtask_id >= len(self.subtasks):
                    # set the next date
                    to_be_scheduled = True
                    scheduled_tasks += 1
                # POSSIBILITY 2 task is scheduled in the future and not completed
                elif not Task.is_subtask_overdue(self.subtasks[subtask_id]) and not self.subtasks[subtask_id]['completed']:
                    # for after_completion tasks these have to be rescheduled only if at least one has been overdue
                    if self.recurring['scheduling'] == 'after_completion':
                        if reschedule:
                            to_be_scheduled = True
                        scheduled_tasks += 1
                    elif self.recurring['scheduling'] == 'in_general':
                        # can stay like it is
                        scheduled_tasks += 1
                    else:
                        print("Error: Unknown scheduling type in add_to_solver")
                # POSSIBILITY 3 task is overdue and not completed
                elif Task.is_subtask_overdue(self.subtasks[subtask_id]) and not self.subtasks[subtask_id]['completed']:
                    # only mandatory, overdue tasks shall be rescheduled
                    if self.overdue_behaviour == 'mandatory':
                        to_be_scheduled = True
                        scheduled_tasks += 1
                # POSSIBILITY 4 task is already marked as completed
                elif self.subtasks[subtask_id]['completed']:
                    pass
                # POSSIBILITY 5 something i have not thought of has happened:
                else:
                    print("Error: add_to_solver ended up in POSSIBILITY 5")

                # if this subtask has been set to be scheduled add it to the optimizer now
                if to_be_scheduled:
                    # TODO: check if task is still in calendar -> delete in that case
                    next_date = None
                    if self.recurring['scheduling'] == 'in_general' and self.overdue_behaviour == 'mandatory':
                        pass
                    elif self.recurring['scheduling'] == 'in_general' and self.overdue_behaviour == 'optional':
                        pass
                    elif self.recurring['scheduling'] == 'after_completion' and self.overdue_behaviour == 'mandatory':
                        pass
                    else:
                        print("ERROR: unallowed combination of scheduling and overdue_behaviour")


                    opt_start = solver.IntVar(0, minimum + 1440000, "opt_start_{number}"
                                              .format(number=task_number))
                    opt_duration = solver.IntVar(0, 1440, "opt_duration_{number}".format(number=task_number))
                    for constraint in self.constraints:
                        constraint.add_constraint(solver=solver,
                                                  time_opt_var=opt_start,
                                                  duration_opt_var=opt_duration,
                                                  relaxation=relaxation)

                    # overdue tasks reaching this point will be rescheduled without a date constraint
                    # otherwise the next date is calculated
                    if next_date is not None and not overdue:
                        if next_date > datetime.date.today():
                            DatespanConstraint(next_date.day, next_date.month, next_date.year,
                                               next_date.day, next_date.month, next_date.year) \
                                .add_constraint(solver=solver,
                                                time_opt_var=opt_start,
                                                duration_opt_var=opt_duration,
                                                relaxation=relaxation)

                    # has to start after now
                    solver.Add((minimum < opt_start ) + (opt_start == 0)*(opt_duration == 0) >= 1)

                    optimization = {'start': opt_start,
                                    'duration': opt_duration,
                                    'task': self,
                                    'subtask_number': subtask_id}
                    optimizations.append(optimization)

                    if subtask_id >= len(self.subtasks):
                        self.subtasks.append({'identifier': None, 'completed': False,
                                              'current_timestamp': None, 'current_duration': None})

                subtask_id += 1











            last_completed_date = None
            last_scheduled_date = None
            completed = len(self.get_completed_subtasks())
            next_date = None
                if subtask_id < len(self.subtasks):
                    # check already created subtasks
                    if self.subtasks[subtask_id]['completed']:
                        # ignore completed task
                        last_completed_date = Task.get_subtask_datetime(self.subtasks[subtask_id]).date()
                    elif self.subtasks[subtask_id]['identifier'] is not None and Task.is_subtask_overdue(self.subtasks[subtask_id]):
                        print("Error: add_to_solver: non-completed, scheduled, overdue task found")
                        exit()
                    elif Task.is_subtask_overdue(self.subtasks[subtask_id]):
                        # non completed and overdue task
                        # only if task is mandatory, the task has to be shifted forward, ignoring day constraints
                        if self.overdue_behaviour == 'mandatory':
                            # TODO: if not scheduled today following tasks will be off, better handling?
                            last_scheduled_date = datetime.date.today()
                            task_number += 1
                            scheduled_tasks += 1
                    else:  # scheduled task which is not overdue
                        if scheduled_tasks == 0:
                            # the first subtask is still not overdue -> all others will be fine as well
                            # skip to the end of the subtasks and only add more tasks if necessary
                            # TODO: check for right spacing (can be off if a task in the future has been deleted)
                            last_scheduled_date = Task.get_subtask_datetime(self.subtasks[-1]).date()
                            scheduled_tasks = len(self.subtasks) - subtask_id
                            subtask_id = len(self.subtasks)-1
                        elif self.recurring['scheduling'] == 'in_general':
                            # there was a subtask which has been rescheduled
                            # for in general scheduling the other tasks can stay as they are
                            last_scheduled_date = Task.get_subtask_datetime(self.subtasks[subtask_id]).date()
                            scheduled_tasks += 1
                        elif self.recurring['scheduling'] == 'after_completion':
                            # tasks like this should be rescheduled
                            # as they will be scheduled depending on the estimated first completion
                            next_date = self.get_next_schedule_date(last_scheduled_date)
                            opt_start = solver.IntVar(0, minimum+1440000, "opt_start_{number}"
                                                      .format(number=task_number))
                            opt_duration = solver.IntVar(0, 1440, "opt_duration_{number}".format(number=task_number))
                            for constraint in self.constraints:
                                constraint.add_constraint(solver=solver,
                                                          time_opt_var=opt_start,
                                                          duration_opt_var=opt_duration,
                                                          relaxation=relaxation)
                            if next_date > datetime.date.today():
                                DatespanConstraint(next_date.day, next_date.month, next_date.year,
                                                   next_date.day, next_date.month, next_date.year) \
                                    .add_constraint(solver=solver,
                                                    time_opt_var=opt_start,
                                                    duration_opt_var=opt_duration,
                                                    relaxation=relaxation)
                            optimization = {'start': opt_start,
                                            'duration': opt_duration,
                                            'task': self,
                                            'subtask_number': subtask_id}
                            optimizations.append(optimization)

                            self.subtasks.append()
                            last_scheduled_date = next_date
                            task_number += 1
                            scheduled_tasks += 1
                else:
                    # new subtasks have to be created
                    if scheduled_tasks == 0:
                        # no reference task scheduled yet
                        if last_completed_date is not None:
                            # all scheduled tasks have been completed
                            next_date = self.get_next_schedule_date(last_completed_date)
                        else:
                            # no task ever scheduled
                            next_date = self.recurring['start_date']
                    else:
                        # already a task scheduled -> schedule the task depending on the last task
                        next_date = self.get_next_schedule_date(last_scheduled_date)

                    # if no next_date has been found the task reached its end_date -> no further scheduling
                    if next_date is None:
                        break

                    # check if next date is already passed, if thats the case pydo has been called rarely
                    # new task is then already overdue and should only be scheduled for mandatory tasks
                    if next_date is not None and (next_date > datetime.date.today() or
                                                  self.overdue_behaviour == 'mandatory'):
                        opt_start = solver.IntVar(0, minimum+1440000, "opt_start_{number}"
                                                  .format(number=task_number))
                        opt_duration = solver.IntVar(0, 1440, "opt_duration_{number}".format(number=task_number))
                        if next_date > datetime.date.today():
                            DatespanConstraint(next_date.day, next_date.month, next_date.year,
                                               next_date.day, next_date.month, next_date.year)\
                                .add_constraint(solver=solver,
                                                time_opt_var=opt_start,
                                                duration_opt_var=opt_duration,
                                                relaxation=relaxation)
                        for constraint in self.constraints:
                            constraint.add_constraint(solver=solver,
                                                      time_opt_var=opt_start,
                                                      duration_opt_var=opt_duration,
                                                      relaxation=relaxation)
                        optimization = {'start': opt_start,
                                        'duration': opt_duration,
                                        'task': self,
                                        'subtask_number': subtask_id}
                        optimizations.append(optimization)
                        last_scheduled_date = next_date
                        task_number += 1
                        scheduled_tasks += 1

                subtask_id += 1

        else:
            # TODO: add overdue behaviour for non recurring tasks
            if self.subtasks[0]['identifier'] is None and not self.subtasks[0]['completed']:
                if len([done_optimization for done_optimization in done_optimizations
                        if done_optimization['task'] == self and done_optimization['subtask_number'] == 0]) > 0:
                    return optimizations
                now = datetime.datetime.now()
                minimum = int((now-REFERENCE_TIME).total_seconds()/60)
                opt_start = solver.IntVar(0, minimum+144000, "opt_start_{number}".format(number=task_number))
                opt_duration = solver.IntVar(0, 1440, "opt_duration_{number}".format(number=task_number))
                for constraint in self.constraints:
                    constraint.add_constraint(solver=solver,
                                              time_opt_var=opt_start,
                                              duration_opt_var=opt_duration,
                                              relaxation=relaxation)

                optimization = {'start': opt_start,
                                'duration': opt_duration,
                                'task': self,
                                'subtask_number': 0}

                optimizations.append(optimization)
            else:
                pass

        return optimizations

    @staticmethod
    def get_subtask_datetime(subtask):
        return REFERENCE_TIME + datetime.timedelta(minutes=subtask['current_timestamp'])

    def __unicode__(self):
        return u'{description}: {current_timestamps} {current_durations} {completed} {identifier}'\
            .format(description=self.description,
                    current_timestamps=[subtask['current_timestamp'] for subtask in self.subtasks],
                    current_durations=[subtask['current_duration'] for subtask in self.subtasks],
                    completed=[subtask['completed'] for subtask in self.subtasks],
                    identifier=[subtask['identifier'] for subtask in self.subtasks])

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
        .format(days="rep_amount", weeks="rep_amount", months="rep_amount", years="rep_amount", kind="rep_kind")
    long_duration_no_params = r"((?:\d+)? *(?:d)(ay(s)?)?|(?:\d+)? *(?:w)(eek(s)?)?|" \
                              r"(?:\d+)? *(?:m)(onth(s)?)?|(?:\d+)? *(?:y)(ear(s)?))?"
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
            (description, constraints, recurring, overdue_behaviour) = TaskSpecifierParamType.extract_match(match)
            return Task(description, constraints, recurring, overdue_behaviour)
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

        if match.group("rep_kind") is not None:
            recurring = {}
            if match.group("rep_kind") == 'd' or match.group("rep_kind") == 'w':
                # days or weeks are mostly handled the same with week just being days
                recurring['kind'] = 'days'
                if match.group("rep_kind") == 'w':
                    multiplier = 7
                else:
                    multiplier = 1
                if match.group("rep_amount") is not None:
                    recurring['amount'] = int(match.group("rep_amount")) * multiplier
                else:
                    recurring['amount'] = 1 * multiplier
            elif match.group("rep_kind") == 'm' or match.group("rep_kind") == "y":
                # years and months are handled the same with year just being 12 months
                # the difference to days is that month and year repetition bind to a date rather than a number of days
                # eg. scheduling every month on the 31.03 will create a task every last day of the month
                # problem with 30 days e.g. would be that the actual date drifts off over time
                recurring['kind'] = 'months'
                today = datetime.datetime.utcnow().date()
                if today.day <= 27:
                    # normal day which can be handled the same for every month
                    recurring['date'] = datetime.datetime.utcnow().day
                elif today.month == 1 or today.month == 3 or today.month == 5 or today.month == 7 or today.month == 8\
                    or today.month == 10 or today.month == 12:
                    # month has 31 days
                    recurring['date'] = today.day - 31 - 1
                elif today.month == 2 and today.day >= 28:
                    # month is february, regard 28th and 29th as last day of every month
                    recurring['date'] = -1
                else:
                    # month has 30 days
                    recurring['date'] = today.day - 30 - 1

                if match.group("rep_kind") == 'y':
                    multiplier = 12
                else:
                    multiplier = 1
                if match.group("rep_amount") is not None:
                    recurring['amount'] = int(match.group("rep_amount")) * multiplier
                else:
                    recurring['amount'] = 1 * multiplier



            if match.group("rep_scheduling") is not None:
                if match.group("rep_scheduling")[0] == 'a':
                    # after completion
                    recurring['scheduling'] = 'after_completion'
                elif match.group("rep_scheduling")[0] == "i":
                    # in general
                    recurring['scheduling'] = 'in_general'
                else:
                    print("Error: rep_scheduling was found but with unknown identifier")
            else:
                recurring['scheduling'] = 'in_general'

        else:
            recurring = None

        if match.group("end_day") is not None:
            datespan_constraint = DatespanConstraint(int_or_none(match.group("start_day")),
                                                     int_or_none(match.group("start_month")),
                                                     int_or_none(match.group("start_year")),
                                                     int_or_none(match.group("end_day")),
                                                     int_or_none(match.group("end_month")),
                                                     int_or_none(match.group("end_year")))

            # datespan strings will be interpreted differently depending if task is recurring or not
            if recurring is None:
                # task shall be scheduled once in this datespan
                constraints.append(datespan_constraint)
            else:
                recurring['start_date'] = datespan_constraint.start_date
                recurring['end_date'] = datespan_constraint.end_date
        else:
            if recurring is not None:
                recurring['start_date'] = datetime.datetime.today()
                recurring['end_date'] = None

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

        if match.group("overdue") is not None:
            overdue_behaviour = match.group("overdue")
        else:
            overdue_behaviour = 'mandatory'

        # TODO: think about wether all 4 combinations of overdue_behaviour and recurring scheduling make sense otherwise check for it

        return description, constraints, recurring, overdue_behaviour


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

        nonrecurring_tasks = [task for task in self.tasks if not task.is_recurring()]
        subtasks = [subtask for task in self.tasks for subtask in task.subtasks]

        for subtask in subtasks:
            if subtask['identifier'] is not None and subtask['completed'] is False:
                subtask['completed'] = True

        for cal_event in cal_events:
            # first check if event is a task
            matched_subtask = next((subtask for subtask in subtasks if subtask['identifier'] == cal_event.identifier),
                                   None)
            if matched_subtask is not None:
                # replan overdue tasks
                # TODO: handling of overdue tasks should be done in plan_tasks and add_to_solver
                # TODO: handling for overdue tasks which can't be rescheduled due to constraints
                matched_subtask['completed'] = False
                if (Task.is_subtask_overdue(matched_subtask)):
                    syncer.remove_subtask(matched_subtask)
                    matched_subtask['identifier'] = None
                continue

            event = next((event for event in self.events if event.identifier == cal_event.identifier), None)
            # add new events
            if event is None:
                # TODO: deleted events
                self.events.append(cal_event)
                continue

        # check for completed tasks -> all tasks which have an id and are completed
        for subtask in subtasks:
            if subtask['completed'] and subtask['identifier'] is not None:
                subtask['identifier'] = None

        # replan the tasks
        self.plan_tasks()

        # add the tasks
        # TODO: readd this when functionining optimization
        syncer.add_tasks(self.tasks)

    def plan_tasks(self, relaxation=0, done_optimizations=None):
        """Use constraint optimization to find possible configuration fulfilling every constraint"""
        solver = pywrapcp.Solver("task_optimization")
        if done_optimizations is None:
            done_optimizations = []
        task_number = 0
        optimizations = []
        opt_vars = []
        scheduled_subtasks = [subtask for task in self.tasks for subtask in task.subtasks
                              if subtask['identifier'] is not None]

        # set up the optimization problem
        for task in self.tasks:
            new_optimizations = task.add_to_solver(solver, task_number, done_optimizations, relaxation)

            for optimization in new_optimizations:
                # no overlap with events from google calendar
                opt_start = optimization['start']
                opt_duration = optimization['duration']
                for event in self.events:
                    solver.Add((opt_start + opt_duration <= event.start_timestamp()) +
                               (opt_start >= event.end_timestamp()) +
                               ((opt_start == 0) * (opt_duration == 0)) >= 1)

                # no overlaps with other to be optimized tasks
                for prev_task_id in range(len(optimizations)):
                    solver.Add((opt_start + opt_duration <= optimizations[prev_task_id]['start']) +
                               (opt_start >= optimizations[prev_task_id]['start'] + optimizations[prev_task_id]['duration']) +
                               ((opt_start == 0) * (opt_duration == 0)) >= 1)

                # no overlaps with already scheduled tasks
                for scheduled_subtask in scheduled_subtasks:
                    solver.Add((opt_start + opt_duration <= scheduled_subtask['current_timestamp']) +
                               (opt_start >= scheduled_subtask['current_timestamp'] + scheduled_subtask['current_duration']) +
                               ((opt_start == 0) * (opt_duration == 0)) >= 1)

                # no overlaps with previous optimization runs
                for done_optimization in done_optimizations:
                    timestamp = done_optimization['task'].subtasks[done_optimization['subtask_number']]['current_timestamp']
                    duration = done_optimization['task'].subtasks[done_optimization['subtask_number']]['current_duration']
                    solver.Add((opt_start + opt_duration <= timestamp) +
                               (opt_start >= timestamp + duration) +
                               ((opt_start == 0) * (opt_duration == 0)) >= 1)

                optimizations.append(optimization)

            task_number += len(new_optimizations)

        if len(optimizations) != 0:
            # solve the problem
            # TODO: enforce some timelimit on the optimization
            # see example at https://github.com/google/or-tools/blob/master/examples/python/steel.py

            # first check which are schedulable without others
            for optimization in optimizations:
                db = solver.Phase([optimization['start'], optimization['duration']], solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
                solver.NewSearch(db)
                optimization['schedulable'] = False
                while solver.NextSolution():
                    if optimization['start'].Value() != 0 and optimization['duration'].Value() != 0:
                        optimization['schedulable'] = True
                        break
                solver.EndSearch()

            # try to solve for these
            schedulable_optimizations = [optimization for optimization in optimizations if optimization['schedulable']]
            opt_vars = []
            for optimization in schedulable_optimizations:
                opt_vars.append(optimization['start'])
                opt_vars.append(optimization['duration'])
                solver.Add(optimization['start'] > 0)
                solver.Add(optimization['duration'] > 0)

            new_optimizations = []
            if len(opt_vars) > 0:
                db = solver.Phase(opt_vars, solver.CHOOSE_FIRST_UNBOUND, solver.ASSIGN_MIN_VALUE)
                solver.NewSearch(db)
                if solver.NextSolution():
                    for optimization in schedulable_optimizations:
                        if optimization['schedulable']:
                            subtask_number = optimization['subtask_number']
                            optimization['task'].subtasks[subtask_number]['current_timestamp'] = optimization['start'].Value()
                            optimization['task'].subtasks[subtask_number]['current_duration'] = optimization['duration'].Value()
                            new_optimizations.append(optimization)
                    done_optimizations += new_optimizations
                else:
                    conflict = True
                solver.EndSearch()

            if relaxation >= MAX_RELAXATION:
                # TODO: give some information about the non scheduled tasks
                print("Some tasks could not be scheduled.")
                return

            # call recursively relaxing the problem with each step
            self.plan_tasks(relaxation+1, done_optimizations)

        else:
            return


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
            for subtask in task.subtasks:
                # only add scheduled tasks which are not overdue
                if subtask['current_duration'] is None \
                        or subtask['current_timestamp'] is None \
                        or subtask['identifier'] is not None \
                        or subtask['completed'] \
                        or Task.is_subtask_overdue(subtask):
                    continue

                # get time object for the current task starting time
                # (for now assuming both are in the same timezone always)
                start_time = REFERENCE_TIME + datetime.timedelta(minutes=subtask['current_timestamp'])
                end_time = REFERENCE_TIME + datetime.timedelta(minutes=subtask['current_timestamp']+
                                                                       subtask['current_duration'])
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
                subtask['identifier'] = event['id']

    def remove_subtask(self, subtask):
        self._service.events().delete(calendarId='primary', eventId=subtask['identifier']).execute()


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
    import sys
    pydo(sys.argv[1:], obj={})
