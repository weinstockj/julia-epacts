function return_missing_indices(phenotype)
    return find(isna.(phenotype))
end

# select IND_ID from PED file that are valid
function return_valid_ped_ids(phenotype, sample_ids)
    missing_indices = return_missing_indices(phenotype)
    non_missing_indices = setdiff(1:length(phenotype), missing_indices)
    return sample_ids[non_missing_indices]
end

# select samples in vcf file where phenotype is not null
function return_valid_vcf_ids(ped_ids, vcf_ids)
    indices = Int64[]
    for id in ped_ids
        index = find(vcf_ids .== id)
        if length(index) == 1
            push!(indices, index[1])
        end
    end
    return indices
end
