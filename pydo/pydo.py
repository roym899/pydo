import abc
import datetime
import httplib2
import os
import yaml
import click

from oauth2client.file import Storage
from oauth2client import client
from oauth2client import tools
from apiclient import discovery


class Constraint:
    """Abstract Base class for all constraints """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_constraint(self):
        pass


class Task:
    """Representation of a task """
    def __init__(self, description, constraints):
        self.description = description
        self.constraints = constraints
        pass


class TaskSpecifierParamType(click.ParamType):
    name = 'task specifier'

    def convert(self, value, param, ctx):
        try:
            # TODO: use regex to scan the task specifier
            return Task("Workout", [])
        except ValueError:
            self.fail('%s is not a valid task specifier' % value)

TASKSPECIFIER = TaskSpecifierParamType()


class Planner:
    """Main class organizing the planning of the tasks """
    def __init__(self):
        self.tasks = []
        self.events = []

    def add_task(self, task):
        self.tasks.append(task)


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

    def get_events(self):
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        print('Getting the upcoming 10 events')
        events_result = self._service.events().list(
            calendarId='primary', timeMin=now, maxResults=2, singleEvents=True,
            orderBy='startTime').execute()

        # print(eventsResult)
        # print(json.dumps(events_result, sort_keys=True, indent=4))

        events = events_result.get('items', [])

        if not events:
            print('No upcoming events found.')
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            print(start, event['summary'])


def load_pydo_data():
    """ Loads the pydo.yaml from ~/.pydo, and returns Planner object stored there"""
    home_dir = os.path.expanduser('~')
    pydo_user_dir = os.path.join(home_dir, '.pydo')
    if not os.path.exists(pydo_user_dir):
        os.makedirs(pydo_user_dir)
    pydo_user_path = os.path.join(pydo_user_dir, 'pydo.yaml')
    try:
        with open(pydo_user_path, 'r') as pydo_file:
            return yaml.load(pydo_file)
    except IOError:
        return Planner()


def save_pydo_data(planner):
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
        print("Pydo without arguments will sync your data with google calendar, use --help for more information.")
        print("Load pydo data...")

        ctx.obj['planner'] = load_pydo_data()

        # TODO: Sync with google calendar and show agenda
        print("TODO: Sync with google calendar...")

        print("Save pydo data...")
        save_pydo_data(ctx.obj['planner'])
    else:
        print("Load pydo data...")
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
