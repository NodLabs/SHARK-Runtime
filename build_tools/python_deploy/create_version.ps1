param(
    [Parameter(Mandatory=$True, Position=0, ValueFromPipeline=$false)]
    [System.String]
    $suffix,

    [Parameter(Mandatory=$True, Position=1, ValueFromPipeline=$false)]
    [System.String]
    $version

)

Write-Host $suffix
Write-Host $version

# Create version info object
$verinfoprop = @{
    'package-suffix'= $suffix
    'package-version'= $version
    'iree-revision'= $(git rev-parse HEAD)
}

$info = New-Object -TypeName PSObject -Prop $verinfoprop

# Convert info to JSON
$info = $info | ConvertTo-JSON

# Output to JSON file
$info | Out-File "version_info.json" -Encoding "UTF8"
