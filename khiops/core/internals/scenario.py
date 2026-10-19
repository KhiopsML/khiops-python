######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes for creating Khiops scenario files"""
import re
import warnings


class ConfigurableKhiopsScenario:
    """A configurable Khiops scenario

    This class encapsulates a template Khiops scenario.
    No parameter replacement is performed at the library level, Khiops will be in charge

    Parameters
    ----------
    template : str
        The template scenario.

    Attributes
    ----------
    template : str
        The template scenario file.
    """

    def __init__(self, template):
        """See class docstring"""
        # Save the template,
        # no need to parse any longer the template as it is already Khiops compliant
        # and the parameter passing is deferred to the very last moment
        self.template = template
        # the keywords are extracted to perform basic checks
        self._parse_keywords()

    def _parse_keywords(self):
        # Initialize private fields
        # keywords outside any loop block
        self._keywords = []
        self._loop_blocks = {}

        loop_begin_re = re.compile(r"^LOOP\s+(__\w+__)")
        loop_end_re = re.compile(r"^END LOOP")
        any_keyword_re = re.compile(r"(__\w+__)")

        # Iterate over the template lines
        line_iter = iter(self.template.split("\n"))
        line = next(line_iter, None)
        line_num = 1
        current_loop_block_name = None
        while line is not None:
            match_loop_begin = loop_begin_re.fullmatch(line)
            if match_loop_begin:
                # Inside a loop block
                current_loop_block_name = match_loop_begin.group(1)
                self._loop_blocks[current_loop_block_name] = []
            elif loop_end_re.fullmatch(line):
                # End of a look block
                current_loop_block_name = None
            else:
                keyword_match = any_keyword_re.search(line)
                if keyword_match is not None:
                    if current_loop_block_name:
                        # Keyword inside a loop block
                        self._loop_blocks[current_loop_block_name].append(
                            keyword_match.group(1)
                        )
                    else:
                        # Keyword outside a loop block
                        self._keywords.append(keyword_match.group(1))
            # Update the template line
            line = next(line_iter, None)
            line_num += 1

    def __str__(self):
        return str(self.template)

    def check_keyword_completeness(self, call_keywords):
        """Check if a list of keys is exactly that of the template of this scenario"""

        # Transform keywords to set
        call_keyword_set = set(call_keywords)
        template_keyword_set = set(self._keywords)

        # Check if there are unknown keys in the input keywords,
        # there should be highlighted but no exception must be raised
        extra_call_keyword_set = call_keyword_set - template_keyword_set
        if extra_call_keyword_set:
            warnings.warn(
                "Call parameter keyword(s) not in template keywords '"
                + ",'".join(extra_call_keyword_set)
                + "'"
            )

        # Check if there are missing keys in the input keywords
        extra_template_keyword_set = template_keyword_set - call_keyword_set
        # HACK : constructionRulesNotEmpty is an added parameter
        extra_template_keyword_set.discard("__constructionRulesNotEmpty__")
        if extra_template_keyword_set:
            raise ValueError(
                "Template parameter keyword(s) not in call keywords '"
                + ",'".join(extra_template_keyword_set)
                + "'"
            )

    def write(self, writer):
        """Writes the template scenario as it is

        Parameters
        ----------
        writer : KhiopsOutputWriter
            The output writer for the scenario file.
        """
        writer.write(self.template)
