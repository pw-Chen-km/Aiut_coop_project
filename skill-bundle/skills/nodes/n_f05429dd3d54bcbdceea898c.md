# USER INTERFACE

This section provides a comprehensive overview of the tablet application's user interface, covering the login process, main navigation bar, and specific operational views. It details the functionality of key menus including Kanbans management, Simple View monitoring, station selection, manual AGV control, and the Node-RED extension tab.
content_type: description
question_intents: How to navigate and operate the tablet application interface

## `n_911df95585dd9a3670727eef` Login Page

This section details how to access the tablet application login page by constructing the correct server URL. It further explains the login process using Qursor or Azure AD credentials and describes the User Menu options for managing language, theme, and session.
content_type: instruction
question_intents: How to access and log in to the tablet application

## `n_32d7775d5ef8c37a23ec568c` Main Menu

The Main Menu section describes the tablet application's primary interface, which is accessible after login and features a top navigation bar. This bar includes an information panel displaying system details like plant name and timestamp, as well as a dropdown menu providing access to various operational views and user settings.
content_type: Instructional
question_intents: How to access and use the main menu and navigation bar

## `n_33e85292bfd76ed855e99522` Kanbans

The Kanbans menu provides a centralized interface for managing production Kanbans through dedicated tabs for filtering, viewing current status, and triggering execution. It includes specific controls for selecting Kanbans via groups or hashtags, monitoring running processes, and modifying their status or details.
content_type: UI Description
question_intents: How to manage and trigger production Kanbans

## `n_5aa064125f98acd23e451191` Simple View

Simple View is a minimal interface within the Qursor system that displays lock zones and station groups for quick monitoring. It presents the names, current states, and assigned tasks of stations within configured groups to facilitate easy reference.
content_type: description
question_intents: What is Simple View?

## `n_25d385d2084244466af01dc1` Stations Choice Menu

The Stations Choice Menu provides an overview of all stations within a plant project, accessible via the navigation bar. It includes tabs for searching and selecting specific stations, monitoring and controlling station statuses, and viewing current or historical task logs.
content_type: navigation
question_intents: How to manage and monitor stations

## `n_d1db741284946631af1c1245` Manual control menu

The manual control menu allows users to add physical AGVs by scanning a QR code via a tablet application, with the specific QR code located in the qursor Configurator application under the AGVs tab.
content_type: instruction
question_intents: How to add a physical AGV using the manual control menu?

## `n_4729bb842358030eb70f21d1` Nodered UI Tab

The Nodered UI tab in the tablet app provides access to additional Node-RED-based features, allowing system expansion without modifying the main app. Access depends on permissions configured in qursor-configurator-application, and the interface can display data such as PLC signal status.
content_type: Documentation
question_intents: What is the Nodered UI tab?
