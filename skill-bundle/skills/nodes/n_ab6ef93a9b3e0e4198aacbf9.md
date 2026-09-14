# Global settings configuration

This section covers the configuration of global system settings, including defining the plant name via the projectName option and managing user account access through group memberships. It also details the setup for Single Sign-On (SSO) integration with Azure Active Directory, specifying the required tenant and application IDs, server address, and HTTPS security requirements.
content_type: configuration
question_intents: How to configure global settings

## `n_da782a308d8967e45aa1b6f9` User account configuration

The user must be a member of a user group with tablet application access configured in qursor-configurator-application.
content_type: instruction
question_intents: How to configure user account access

## `n_bae2f24b58e420a19d9d9a1e` SSO configuration

The section describes the option to use SSO with Azure Active Directory, requiring configuration at system installation and enabling in frontend settings by providing the Azure tenant ID, application ID, and server address. It specifies the format for the azureSettings section and mandates the use of HTTPS with a valid SSL certificate on the Qursor server for the integration to function.
content_type: configuration
question_intents: How to configure SSO with Azure Active Directory
