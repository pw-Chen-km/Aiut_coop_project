# Qursor-MAP-userguide-Kanbans-20250121

This user guide provides comprehensive instructions for the Qursor M.A.P. interface, covering initial login, application settings, and the main dashboard for monitoring AGVs, stations, and tasks. It details operational workflows for task execution, AGV diagnostics, and issue management, with a significant focus on the definition, configuration, execution, and production application of Kanbans.
content_type: User Guide
question_intents: How to use Qursor M.A.P. for AGV management and Kanban configuration
aliases: Qursor M.A.P. User Guide

## `n_5efdab0844a7b4124a2ac028` Qursor M.A.P. User Guide

The section provides the document version date as administrative information.
content_type: administrative_information

## `n_81e57ecf2f0afbffc587b2b2` Quick start

The section lists topics including logging in, application settings, dashboard, task realization, AGV diagnostics, tasks overview, stations overview, diagnostic messages, issues, Kanbans, and aiut.
content_type: table_of_contents

## `n_d6a21616641921a92711d896` First logging in

The section instructs users to enter their user name as the password during the first login and then enter and confirm a new password.
content_type: instruction
question_intents: How to log in for the first time

## `n_4893f402d5dd1a7bffe09bbc` Application settings

This section outlines the Qursor Configurator interface, detailing how to switch between automatic and autonomous system modes and access user-specific preferences such as language and session management. It also covers the application's core purpose of defining stations, tasks, and AGVs, along with instructions for verifying the current software version.
content_type: description
question_intents: How to manage application settings and system modes

## `n_7667a436070281a7c7919684` Main dashboard

The main dashboard, accessed after login, displays a map view and real-time information on stations, AGVs, and tasks. It is divided into four areas: Locations, Map, AGVs, and Tasks, each providing specific details and controls for monitoring and managing operations.
content_type: User Interface Documentation
question_intents: What are the main areas of the dashboard?

## `n_e4dd86545d07aabd0b7126a8` Main dashboard - Locations

This section defines AGV locations as specific points for loading, unloading, or parking, including types such as chargers and wait points. It details the dashboard interface for managing these locations, covering actions like viewing assets, checking status indicators, and modifying settings such as adding conveyors or disabling stations.
content_type: UI Documentation
question_intents: How to manage locations in the dashboard

## `n_9803bc1dc40a7f3d3f5fe633` Main dashboard - Map

The map is a central element of the Dashboard view that displays AGV positions, paths, and stations. Users can navigate by dragging, zooming, and rotating the view. The map supports optional zones, such as lock and traffic light zones, and indicates station statuses (enabled, disabled, or in operation) via color coding. Additional features include visualizing navigation scanners, heat maps, saving views, and adjusting layers.
content_type: User Interface Documentation
question_intents: How to interact with the map

## `n_69a5c70ec1b7028efe029acc` Main dashboard - AGV

The section describes the main dashboard for AGVs, detailing indicators for loading, battery, and operational status (such as Automatic, On Hold, or Disconnected). It outlines available controls for manual positioning, blackbox viewing, and specific actions like pausing, resetting paths, or managing resources at stations, noting that certain options are exclusive to disabled AGVs.
content_type: UI Documentation
question_intents: How to control AGV status and actions

## `n_ab20276207cd4831fb75b9d2` Task performance 1/3

This section outlines the procedure for executing tasks at stations configured in the Configurator application. It details the workflow of selecting a station in the Dashboard to trigger an AGV call, reserve the station, and visualize the destination path on the map.
content_type: procedural_guide
question_intents: how_to_run_task

## `n_447e1d4f88ed44ffff2cbfb6` Task performance 2/3

The section describes the sequence where an AGV arrives at station 3.6, is loaded, and then proceeds to station 3.5, which is locked and waiting for the AGV.
content_type: Process Description
question_intents: AGV Loading and Routing

## `n_f4f0efb9353e3efd003078e8` Task performance 3/3

The section reports that a loaded AGV has arrived at station 3.5, the task is complete and removed from the active list, and no tasks are currently assigned to any stations.
content_type: Status Report
question_intents: Task Completion Status

## `n_d15e16c206e8b6ff7a4ea849` Diagnostic of AGV

The section describes the AGV diagnostic tab, which allows users to search and sort AGVs by name and access detailed diagnostic data. It includes a blackbox preview showing sensor views, position, and events, as well as a detailed information view displaying map position, battery levels, errors, and system messages.
content_type: User Interface Documentation
question_intents: How to view AGV diagnostic data
aliases: diagnostic

## `n_16c9ef8d9217c6840e99e78a` Tasks overview

This section defines AGV tasks as station-triggered operations and outlines the interface for managing them, including viewing current and archived entries and removing in-progress tasks. It details the specific data fields associated with active and archived tasks, such as trigger time, destination, and final results, while distinguishing between the executable current view and the immutable archive.
content_type: Documentation
question_intents: What is a task?

## `n_8b33aaae099b66966c91cade` Stations overview

This section defines stations as sites for AGV loading, unloading, and parking, detailing their basic parameters such as ID, Map ID, Name, and Reserved status. It also describes the user interface elements that display the AGV load state, current task, and presence status at these stations.
content_type: Reference
question_intents: What are the basic parameters and status indicators for AGV stations?

## `n_ff48e981df0c457fd7318e0b` Stations overview - more data

This section details the status data and task types displayed in the Stations list, including how to switch between autonomous and automatic modes. It also explains how to access the list of finished tasks and view the details of the task currently executing at the station.
content_type: description
question_intents: How to view executed tasks

## `n_06d52deae093e151367c3da7` Diagnostic messages

This section defines diagnostic message components, including module abbreviations such as QFM and OM, and severity levels like Information, Error, and Warning.
content_type: Reference
question_intents: What do the diagnostic message components mean?

## `n_66f294b90e54edefe4e3d1fa` Issues

The Issues section defines user-reported problems and details the interface for viewing, filtering, searching, and exporting them, including status indicators and actions for reporting, editing, or removing entries. It is accompanied by administrative version information for the Qursor M.A.P. User Guide.
content_type: Documentation
question_intents: How to manage reported issues

## `n_b6e253da6059fe229fa3c12c` Kanbans

The section provides an overview of Kanbans, including their definition, possible step statuses, configuration options, editing procedures, sequence settings, application to production, relations settings, execution details, reservation areas, and import/export capabilities.
content_type: Documentation
question_intents: What are Kanbans?

## `n_ee92a0e810727f4a2af53941` What are Kanbans?

Kanbans are composed of steps containing sequenced jobs with optional preconditions that trigger use cases, and they can be initiated by operators or external systems via the Qursor API.
content_type: definition
question_intents: definition

## `n_65c36cb6d4621a50fda97843` Kanbans overview

The section describes the interface elements of the Kanbans overview list, including controls for dragging, editing, and deleting Kanbans, as well as fields displaying the Kanban name, current step, status, assigned AGVs, and required stations.
content_type: UI Reference
question_intents: Kanban list interface elements

## `n_a2d2e3d5ca86eff3618e2963` Kanban execution 1/2

The section explains how to access the currently executed sequence of a Kanban by clicking the down arrow next to the selected Kanban. It describes the visual indicators for job status, such as green marking for the active job and red lock icons for rejected reservations. Additionally, it outlines the functions of the flash icon options, including Skip Job, Force, and Go To, which allow users to bypass conditions or switch jobs directly.
content_type: Instructional Guide
question_intents: How to view Kanban execution details

## `n_83c4a82307432c148c86cc5f` Kanban execution 2/2

This section details the prioritization logic for AGV/AMR vehicle reservations during Kanban step execution. It explains how the reservation queue is ordered based on priority settings, time to empty, addition order, and manual adjustments, with reservations attempted sequentially from the top of the queue.
content_type: procedural
question_intents: How are AGV/AMR vehicle reservations prioritized for Kanban execution?

## `n_1777ee7409cff7a05431f83b` Kanbans overview - possible statuses of Kanban steps

The section lists the possible statuses of Kanban steps, including On hold, Preparing, Waiting for address, Queued, Executing, Step completed, Finished, Cancelling, Cancelling completed, and Deleting, along with their associated color markings and descriptions.
content_type: Reference
question_intents: What are the possible statuses of Kanban steps?

## `n_dffc02aea8c54ddbfc5e984e` Kanban configuration - options overview 1/2

This section outlines the initial steps for configuring Kanbans by accessing the Qursor Configurator application from within Qursor M.A.P. It details the specific navigation path required to open the Kanban options menu.
content_type: procedural
question_intents: How to access Kanban configuration

## `n_bcdeffa9af0ded4ad9052b89` Kanban configuration - options overview 2/2

This section provides an overview of eight specific options within the Kanban configuration interface, including settings, searching, applying to production, relations, reservation settings, custom parameters, import/export, adding new Kanbans, and the list of available Kanbans.
content_type: overview
question_intents: What are the options in the Kanban configuration interface?

## `n_d170956078dcc3a69b89db4b` Kanban editing 1/2

The section lists eight numbered interface elements for Kanban editing, describing functions such as version selection, status indicators, station replacement, copying, resetting, saving, and creating new versions.
content_type: UI Reference
question_intents: Kanban version management

## `n_a2cbf66f4d97ce230be3bc8e` Kanban editing 2/2

This section details the configuration of Kanban relations to regulate workflow between steps and sequence jobs, as well as the setup of allowed dynamic stations for real-time station definition. It covers how to define specific stations that external systems can use to replace configured dynamic stations when triggering a Kanban.
content_type: Configuration Reference
question_intents: How to configure Kanban settings, relations, and dynamic stations

## `n_93f9f900e27049e101be935e` Kanban steps

This section outlines the procedures for adding, configuring, reordering, and removing steps in a Kanban board, including the definition of basic configuration fields such as Name, Symbol, Priority, and Hashtags. It also details the management of station reservations for AGVs, covering the addition, selection, prioritization, and removal of stations to determine vehicle requirements.
content_type: Instructional
question_intents: How to manage Kanban steps and station reservations

## `n_631dd76e59b3dede8485ba36` Sequence configuration 1/2

The section defines a sequence as a specific order of jobs in a Kanban step, outlining progression from initial to final states. It describes the graphical visualization of jobs, conditions, and use cases, and lists available actions such as editing, removing, adding, and describing jobs. It also details the roles of START and END jobs and the meaning of black and red arrows indicating conditional flow.
content_type: Documentation
question_intents: How to configure a Kanban sequence

## `n_289fb690557d8d2f38826c13` Sequence configuration 2/2

This section guides users on editing sequence jobs by selecting a job and entering new parameters in the Edit sequence job window. It details the configuration of the next job, including conditional triggers for when conditions are met or not met. Additionally, it defines specific trigger parameters such as station, destination, AGV type, and execution conditions based on station status, items, dynamic stations, or custom parameters.
content_type: instruction
question_intents: How to edit and configure sequence job parameters and triggers

## `n_c4c72b1b552f28c563f58729` Kanbans applied for production - overview

The section provides an overview of Kanbans applied for production, detailing the current list, configuration statuses (CURRENT/NEW), and the download status of the Connector and Manager modules (DONE/REQUEST). It also includes information on the user who downloaded the configuration, creation and completion dates, and an archive of previously applied configurations.
content_type: overview
question_intents: What is the status of Kanban configurations applied to production?

## `n_4bb82c4495ede8f744982b6c` Applying Kanbans to production 1/2

The section describes the steps to apply Kanbans to production, including adding a new configuration, viewing available versions, modifying settings, and saving changes.
content_type: procedural
question_intents: How to apply Kanbans to production

## `n_186f2ec1dbfffb1330f9639b` Applying Kanbans to production 2/2

The section describes the process of applying Kanban changes to production, detailing change types (NEW, CHANGED, DELETED, TO DELETED), version selection, and specific action buttons such as Save and Apply to production.
content_type: documentation
question_intents: How to apply Kanban changes to production

## `n_c27ee75e540c116028ea1706` Kanban relations

Kanban relation groups define which Kanbans cannot be executed simultaneously by checking for concurrent starts within the same group; if a conflict occurs, the name of the blocking Kanban is displayed next to the status of the blocked Kanban.
content_type: explanation
question_intents: how do kanban relation groups prevent simultaneous execution

## `n_7b9c77009d36ab830c846565` Kanban relations - settings

The section describes how to view, add, and edit relation groups and Kanbans, including changing steps and sequence jobs to manage blocking behavior, and explains how configurations are displayed based on current or new production versions.
content_type: Instructional
question_intents: How to configure Kanban relations

## `n_6caaafe4fe4aa11a71f3e883` Reservation areas

The section describes reservation areas as a mechanism for reserving AGV robots for Kanban jobs, detailing how Kanbans are assigned to specific areas or a default queue, and outlining the interface actions for adding new areas and viewing available ones.
content_type: Documentation
question_intents: How do reservation areas manage AGV queues?

## `n_a0c3a41196644da72439be27` Custom parameters

Custom parameters are additional inputs passed by external systems to Kanban tasks. Users can add new parameters by clicking the plus icon and remove them using the trash icon.
content_type: Instructional
question_intents: How to manage custom parameters

## `n_0db6fdaad61be6f9cb29ca79` Import/Export

This section covers the procedures for exporting Kanbans as .zip files and the requirements for importing them, including necessary data fields and available actions. It details how to select items and versions for export, as well as the specific inputs and default behaviors involved in the ingestion process.
content_type: instructions
question_intents: How to import and export Kanbans
aliases: Kanban Data Transfer
