function[vecs] = string_to_vecs(strng, dict)
% input: query string and dictionary mapping of words to 300 dim vectors
% output; matrix of size 300x num_words with w2v vectors if word is in dict
%         0 vector otherwise

    words = sanitize_string(strng);
    vecs = zeros(300, length(words));
    for i = 1:length(words)
        if isKey(dict, words{i})
            vecs(:,i) = dict(words{i});
        end            
    end
end

