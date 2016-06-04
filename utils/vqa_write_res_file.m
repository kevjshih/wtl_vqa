function[] = vqa_write_res_file(res_map, res_out_file)
%% Function to convert network output in map form to expected json file
    all_qs = keys(res_map);
    fid = fopen(res_out_file, 'w');
    fprintf(fid,'[');
    for i = 1:length(all_qs)
        if mod(i, 1000) == 0
            fprintf('%d/%d\n', i, length(all_qs));
        end
        ans_str = res_map(all_qs{i});
        ans_str = regexprep(ans_str, '\\', '\\\\');
        ans_str = regexprep(ans_str, '"', '\\"');
        fprintf(fid,'{"answer": "%s", "question_id": %d}', ans_str, all_qs{i});
        if i < length(all_qs)
            fprintf(fid,', ');
        end
    end
    fprintf(fid,']');
    fclose(fid);


end