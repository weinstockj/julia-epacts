function return_missing_indices(phenotype::DataArrays.DataArray{Int64, 1})
    return find(isna.(phenotype))
end

# select IND_ID from PED file that are valid
function return_valid_ped_ids(phenotype::DataArrays.DataArray{Int64, 1}, 
                              sample_ids::DataArrays.DataArray{String, 1})
    missing_indices = return_missing_indices(phenotype)
    non_missing_indices = setdiff(1:length(phenotype), missing_indices)
    return sample_ids[non_missing_indices]
end

# select samples in vcf file where phenotype is not null
function return_valid_vcf_ids(ped_ids::DataArrays.DataArray{String, 1}, 
                              vcf_ids::Array{String, 1})
    indices = Int64[]
    for id in ped_ids
        index = find(vcf_ids .== id)
        if length(index) == 1
            push!(indices, index[1])
        end
    end
    return indices
end

function trim_and_convert_phenotype(phenotype::DataArrays.DataArray{Int64, 1})
    return convert(Array, phenotype[~isna(phenotype)])
end

function extract_covars(ped::DataFrames.DataFrame, phenotype::DataArrays.DataArray{Int64, 1})
    covars = ped[[:SEX, :AGE, :PC1, :PC2, :PC3, :PC4]]
    missing_indices = return_missing_indices(phenotype)
    non_missing_indices = setdiff(1:length(phenotype), missing_indices)
    return covars[non_missing_indices, :]
end   
