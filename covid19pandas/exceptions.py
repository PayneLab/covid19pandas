#   Copyright 2018 Samuel Payne sam_payne@byu.edu
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# User-directed exceptions
class PackageError(Exception):
    """Base class for all exceptions we'll raise."""
    pass

class NoInternetError(PackageError):
    """No internet."""
    pass

class FileDoesNotExistError(PackageError):
    """The file does not exist."""
    pass

class ParameterError(PackageError):
    """Parameter error."""
    pass

# Warnings
class PackageWarning(UserWarning):
    """Base class for all warnings we'll generate."""
    pass

class OldPackageVersionWarning(PackageWarning):
    """They're using an old version of the package."""
    pass

class FileNotUpdatedWarning(PackageWarning):
    """Data file was not updated."""
    pass

class DeprecatedWarning(PackageWarning):
    """Something is deprecated."""
    pass

# Developer-directed exceptions
class PackageDevError(Exception):
    """For exceptions that are probably the developer's fault."""
    pass
