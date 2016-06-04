function[words] = sanitize_string(str)
% converts string to lower case, splits by spaces, and removes non-alphanumerics except single quotes and dashes
str = lower(strtrim(str));
words = strsplit(str, {' ','\f','\n','\r','\t','\v', '\\', '/', ',', char(39), '-'});

words = regexprep(words,'[^-a-zA-Z0-9'']','');
words = words(~cellfun('isempty', words));


end